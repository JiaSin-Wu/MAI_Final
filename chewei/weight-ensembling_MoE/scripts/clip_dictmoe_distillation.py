from _common import *

from src.adamerging import softmax_entropy
from src.datasets.common import maybe_dictionarize

log = logging.getLogger(__name__)

from collections import defaultdict
from typing import cast

import lightning as L
import open_clip.model
from clip_checkpoint_path_distillation import (
    CHECKPOINT_DIR,
    pretrained_model_path,
)
from pathlib import Path

# Distillation checkpoint directory
DISTILLATION_DIR = Path("/data1/chliu/MAI2025_final/results/clip_distillation/ViT-L-14/checkpoints")
from lightning.fabric.wrappers import _FabricModule
from torch.utils.data import DataLoader

from src.clip_eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.module.dict_moe import DictMoE
from src.module.utils import get_by_name, print_trainable_parameters, set_by_name
from src.task_vectors import StateDict, TaskVector, state_dict_mean
from src.ties_merging_utils import check_parameterNamesMatch
from src.utils import timeit_context 


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.model is None:
            raise ValueError("model must be specified")

        self.result_dir = RESULTS_DIR / cfg.exp_name / cfg.model
        if cfg.version is not None:
            self.result_dir /= f"version_{cfg.version}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        log.info(f'files will save to {self.result_dir}')
        # save `cfg` to result_dir`
        self.result_path = self.result_dir / "results.csv"

        self.fabric = L.Fabric(
            accelerator="cuda",
            devices=cfg.num_devices,
            strategy="ddp"
            # strategy=self._fsdp_strategy() if cfg.model == "ViT-L-14" else "ddp",
        )
        self.fabric.launch()

    def _fsdp_strategy(self):
        policy = {open_clip.model.ResidualAttentionBlock}
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=policy,
            # state_dict_type="full",
            # activation_checkpointing_policy=policy if cfg.model == "ViT-L-14" else None,
        )
        return strategy

    def run(self):
        self.load_model()
        self.load_datasets()

        if self.cfg.tta:
            self.tta()
        if self.cfg.evaluate:
            self.evaluate()

    def tta(self):
        OmegaConf.save(self.cfg, self.result_dir / "tta_config.yaml")

        target_update_steps = 250
        accum_steps = self.cfg.accumulate_grad_batches
        total_loop_steps = target_update_steps * accum_steps

        model = deepcopy(self.model)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=self.cfg.lr)
        model, optimizer = self.fabric.setup(model, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_update_steps)

        (self.result_dir / "checkpoints").mkdir(exist_ok=True)

        model.train()
        for step_idx in tqdm(range(total_loop_steps), "tta training"):
            losses = 0
            for dataset_idx, dataset_name in enumerate(self.cfg.seen_datasets):
                try:
                    batch = next(self.shuffled_test_loader_iters[dataset_idx])
                except StopIteration:
                    # Iterator is exhausted, create a new one
                    self.shuffled_test_loader_iters[dataset_idx] = iter(self.shuffled_test_loaders[dataset_idx])
                    batch = next(self.shuffled_test_loader_iters[dataset_idx])
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(self.fabric.device)  # use images only

                features = model(x)
                logits = self.classification_heads[dataset_name](features)

                loss = softmax_entropy(logits).mean(0)
                losses += loss

            scaled_loss = losses / accum_steps
            self.fabric.backward(scaled_loss)
            if (step_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                print(f"step={step_idx}, loss={losses.item()}")

            # Save checkpoint every 500 steps
            if (step_idx + 1) % 500 == 0:
                self.fabric.save(self.result_dir / "checkpoints" / f"model_step={step_idx + 1}.ckpt", {"model": model})
                log.info(f"Saved checkpoint at step {step_idx + 1}")

        # Save final checkpoint
        self.fabric.save(self.result_dir / "checkpoints" / f"model_step={total_loop_steps}.ckpt", {"model": model})

    @torch.inference_mode()
    def evaluate(self):
        results = defaultdict(list)
        target_update_steps = 250
        accum_steps = self.cfg.accumulate_grad_batches
        total_loop_steps = target_update_steps * accum_steps

        # Only evaluate the final checkpoint
        step_idx = total_loop_steps

        state_dict = torch.load(self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt", map_location="cpu")
        if len(state_dict) == 1 and "model" in state_dict:
            state_dict = state_dict["model"]

        model = deepcopy(self.model)

        # Verify loading works correctly
        info = model.load_state_dict(state_dict, strict=False)
        if info.missing_keys:
            log.warning(f"Step {step_idx}: {len(info.missing_keys)} missing keys during load_state_dict")
            log.warning(f"  First few: {info.missing_keys[:5]}")
        if info.unexpected_keys:
            log.warning(f"Step {step_idx}: {len(info.unexpected_keys)} unexpected keys during load_state_dict")
            log.warning(f"  First few: {info.unexpected_keys[:5]}")

        log.info(f"Total keys in checkpoint: {len(state_dict)}")
        log.info(f"Total keys in model: {len(model.state_dict())}")

        # Verify router weights were actually loaded
        sample_router_key = "model.visual.transformer.resblocks.18.mlp.gate.fc2.bias"
        if sample_router_key in state_dict:
            loaded_value = model.state_dict()[sample_router_key]
            expected_value = state_dict[sample_router_key]
            if not torch.allclose(loaded_value, expected_value):
                log.error(f"Step {step_idx}: Router weights NOT loaded correctly!")
            else:
                log.info(f"Step {step_idx}: Router weights verified - successfully loaded")
                log.info(f"  Initial (self.model): {self.model.state_dict()[sample_router_key]}")
                log.info(f"  Loaded from ckpt:     {loaded_value}")
                # Check if it's different from initial model
                initial_value = self.model.state_dict()[sample_router_key]
                if torch.allclose(loaded_value, initial_value):
                    log.warning(f"  WARNING: Loaded router weights are SAME as initial model!")
                else:
                    log.info(f"  ✓ Router weights are different from initial model (as expected)")

        model = self.fabric.setup_module(model)
        model.eval()
        results["step"].append(step_idx)

        for dataset_idx, dataset_name in enumerate(
            tqdm(
                self.cfg.test_datasets,
                "evaluating datasets",
                leave=False,
            )
        ):
            test_loader = self.test_loaders[dataset_idx]
            TOTAL_CORRECT = 0
            TOTAL_COUNT = 0
            for batch in (
                pbar := tqdm(
                    test_loader,
                    f"evaluate {dataset_name}",
                    leave=False,
                )
            ):
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(self.fabric.device)
                y = batch["labels"].to(self.fabric.device)

                features = model(x)
                logits = self.classification_heads[dataset_name](features)
                preds = logits.argmax(-1)

                # Debug: Check if predictions are valid (only for first batch of first dataset)
                if TOTAL_COUNT == 0 and dataset_idx == 0:
                    log.info(f"Debug for {dataset_name}:")
                    log.info(f"  Features shape: {features.shape}, mean: {features.mean():.4f}, std: {features.std():.4f}")
                    log.info(f"  Logits shape: {logits.shape}, min: {logits.min():.4f}, max: {logits.max():.4f}")
                    log.info(f"  Preds: {preds[:10]}")
                    log.info(f"  Labels: {y[:10]}")
                    log.info(f"  Unique preds: {preds.unique()}")

                correct = (preds == y).sum().item()
                TOTAL_CORRECT += correct
                TOTAL_COUNT += len(y)
                acc = TOTAL_CORRECT / TOTAL_COUNT
                pbar.set_postfix_str(f"acc={acc:.2f}")
            results[dataset_name].append(acc)
        (df := pd.DataFrame(results)).to_csv(self.result_path, index=False)
        log.info(df)

    def load_clip_models(self):
        """
        Loads the pretrained CLIP model and the distilled models from distillation checkpoints.

        The distilled checkpoints already contain a merged backbone (base + task vectors),
        so we load them directly without needing to recompute the merge.

        Side Effects:
            Sets the instance variables `pretrained_model`, `distilled_models`, and `classification_heads`.
        """
        cfg = self.cfg

        # load pretrained and distilled models
        with timeit_context():
            log.info("load models from distillation checkpoints")
            pretrained_model: ImageEncoder = torch.load(pretrained_model_path(cfg.model), map_location="cpu")

            distilled_models: List[ImageEncoder] = []
            for dataset_name in track(
                cfg.seen_datasets if cfg.model_seen_datasets is None else cfg.model_seen_datasets,
                "loading distilled models",
            ):
                distillation_path = DISTILLATION_DIR / f"distillation_{dataset_name}.pt"
                log.info(f"Loading distilled model for {dataset_name} from {distillation_path}")

                if not distillation_path.exists():
                    raise FileNotFoundError(f"Distilled model not found: {distillation_path}")

                # Load distillation checkpoint (state_dict with "model." prefix)
                state_dict = torch.load(distillation_path, map_location="cpu", weights_only=False)

                # Strip "model." prefix from keys
                # Distillation checkpoints have keys like "model.visual.xxx"
                # But ImageEncoder.model expects keys like "visual.xxx"
                stripped_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        new_key = key[len("model."):]  # Remove "model." prefix
                        stripped_state_dict[new_key] = value
                    else:
                        stripped_state_dict[key] = value

                # Create ImageEncoder and load the stripped state_dict
                distilled_model = ImageEncoder(cfg, keep_lang=False)
                missing_keys, unexpected_keys = distilled_model.model.load_state_dict(stripped_state_dict, strict=False)

                if missing_keys:
                    log.warning(f"Missing keys for {dataset_name}: {len(missing_keys)}")
                if unexpected_keys:
                    log.warning(f"Unexpected keys for {dataset_name}: {len(unexpected_keys)}")

                distilled_models.append(distilled_model)
                log.info(f"  ✓ Loaded {dataset_name}")

        self.pretrained_model = pretrained_model
        self.distilled_models = distilled_models
        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).eval() for dataset_name in cfg.test_datasets}
        for m in self.classification_heads.values():
            for p in m.parameters():
                p.requires_grad_(False)
        self.classification_heads = {k: m.to(self.fabric.device) for k, m in self.classification_heads.items()}

    @torch.no_grad()
    def load_model(self):
        self.load_clip_models()
        with timeit_context("Building moe model"):
            # Use the first distilled model as the backbone
            # All distilled models have identical non-MLP layers (merged backbone)
            # We just need to replace the MLPs with DictMoE
            model = deepcopy(self.distilled_models[0])
            log.info("Using merged backbone from distilled models (no fusion needed)")

            # fix all parameters
            for p in model.parameters():
                p.requires_grad_(False)

            merging_mlp_layer = self.cfg.get("DictMoe_layer")
            if merging_mlp_layer is None:
                merging_mlp_layer = list(range(model.model.visual.transformer.layers))
                log.info("`DictMoe_layer` not specified, default to all layers.")

            for layer_idx in range(model.model.visual.transformer.layers):
                if layer_idx not in merging_mlp_layer:
                    continue

                log.info(f"Replacing MLP of layer {layer_idx} with DictMoE.")
                # Use pretrained base MLP and distilled expert MLPs
                model.model.visual.transformer.resblocks[layer_idx].mlp = DictMoE(
                    hidden_size=model.model.visual.transformer.width,
                    base_model=self.pretrained_model.model.visual.transformer.resblocks[layer_idx].mlp,
                    expert_models=[m.model.visual.transformer.resblocks[layer_idx].mlp for m in self.distilled_models],
                    init_lambda=self.cfg.init_lambda,
                    fix_base_model_and_experts=True,
                    router_hidden_layers=self.cfg.router_hidden_layers,
                )

            self.model = model
            print_trainable_parameters(model, verbose=True)

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration.

        It first imports the necessary modules and sets up a basic transform for the images.
        It then loads each dataset specified in the configuration, applies the basic transform,
        and sets the location, batch size, and number of workers from the configuration.

        The test dataset from each loaded dataset is added to the list of test datasets.
        It then sets up the data loaders for the test datasets, both with
        and without shuffling, and creates an iterator for each shuffled test loader.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        cfg = self.cfg
        cfg.batch_size = cfg.batch_size // cfg.num_devices
        cfg.tta_batch_size = cfg.tta_batch_size // cfg.num_devices
        cfg.eval_batch_size = cfg.eval_batch_size // cfg.num_devices
        print(f"batch_size={cfg.batch_size}, tta_batch_size={cfg.tta_batch_size}, eval_batch_size={cfg.eval_batch_size}")

        if self.cfg.corruption is None:
            from src.datasets.registry import get_dataset
        else:
            from src.datasets.corruption.registry import get_dataset

        cfg = self.cfg

        dataset_kwargs = dict(
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if self.cfg.corruption is not None:
            dataset_kwargs["corruption"] = self.cfg.corruption
        datasets = [
            get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                **dataset_kwargs,
            )
            for dataset_name in cfg.test_datasets
        ]
        self.test_datasets = [d.test_dataset for d in datasets]
        self.test_loaders = [
            DataLoader(
                d,
                shuffle=False,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loaders = [
            DataLoader(
                d,
                shuffle=True,
                batch_size=cfg.tta_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loader_iters = [iter(d) for d in self.shuffled_test_loaders]


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_dictmoe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    cfg.save = CACHE_DIR / "checkpoints" / "task_vectors_checkpoints" / cfg.model
    # Only set default data_location if not provided via command line
    if cfg.data_location == "???":
        cfg.data_location = str(DATA_DIR)
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()