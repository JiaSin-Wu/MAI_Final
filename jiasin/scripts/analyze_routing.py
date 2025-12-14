from _common import *

import logging

log = logging.getLogger(__name__)

from collections import defaultdict
from functools import partial

from clip_dictmoe import Program
from src.datasets.common import maybe_dictionarize
from src.module.dict_moe import DictMoE


class AnalysisProgram(Program):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.gate_weights_by_layer = defaultdict(list)

    @torch.no_grad()
    def load_model(self):
        super().load_model()

        def hook_fn(module, input, output, layer_idx):
            self.gate_weights_by_layer[layer_idx].append(output.detach().cpu())

        for layer_idx, resblock in enumerate(self.model.model.visual.transformer.resblocks):
            if isinstance(resblock.mlp, DictMoE):
                resblock.mlp.gate.register_forward_hook(partial(hook_fn, layer_idx=layer_idx))

    @torch.inference_mode()
    def evaluate(self):
        # Clear gate weights before evaluation
        for layer_idx in self.gate_weights_by_layer:
            self.gate_weights_by_layer[layer_idx].clear()

        results = defaultdict(list)
        target_update_steps = 250
        accum_steps = self.cfg.accumulate_grad_batches
        total_loop_steps = target_update_steps * accum_steps
        for step_idx in tqdm([total_loop_steps], "evaluating", leave=False):
            state_dict = torch.load(self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt", map_location="cpu")
            if len(state_dict) == 1 and "model" in state_dict:
                state_dict = state_dict["model"]
            model = deepcopy(self.model)
            model.load_state_dict(state_dict)
            model = self.fabric.setup_module(model)
            model.eval()
            results["step"].append(step_idx)

            all_samples_data = []  # List to store (dataset_name, label)

            for dataset_idx, dataset_name in enumerate(tqdm(self.cfg.test_datasets, "evaluating datasets", leave=False)):
                test_loader = self.test_loaders[dataset_idx]
                TOTAL_CORRECT = 0
                TOTAL_COUNT = 0
                for batch in (pbar := tqdm(test_loader, f"evaluate {dataset_name}", leave=False)):
                    batch = maybe_dictionarize(batch)
                    x = batch["images"].to(self.fabric.device)
                    y = batch["labels"].to(self.fabric.device)

                    # Collect dataset_name and labels
                    for label in y.cpu().numpy():
                        all_samples_data.append({"Task": dataset_name, "label": label})

                    features = model(x)
                    logits = self.classification_heads[dataset_name](features)
                    preds = logits.argmax(-1)

                    correct = (preds == y).sum().item()
                    TOTAL_CORRECT += correct
                    TOTAL_COUNT += len(y)
                    acc = TOTAL_CORRECT / TOTAL_COUNT
                    pbar.set_postfix_str(f"acc={acc:.2f}")
                results[dataset_name].append(acc)
            (df := pd.DataFrame(results)).to_csv(self.result_path, index=False)
            log.info(df)

            # Save all_samples_data to samples.csv
            samples_df = pd.DataFrame(all_samples_data)
            samples_df.to_csv(self.result_dir / "samples.csv", index=False, header=False)
            log.info(f"Saved samples data to {self.result_dir / 'samples.csv'}")

            # Save gate weights
            for layer_idx, weights in self.gate_weights_by_layer.items():
                weights_df = pd.DataFrame(torch.cat(weights).mean(dim=1).numpy())
                weights_df.to_csv(self.result_dir / f"gate_layer={layer_idx}.csv", index=False, header=False)
                log.info(f"Saved gate weights for layer {layer_idx} to {self.result_dir / f'gate_layer={layer_idx}.csv'}")


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_dictmoe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    cfg.save = CACHE_DIR / "checkpoints" / "task_vectors_checkpoints" / cfg.model
    cfg.data_location = str(DATA_DIR)
    program = AnalysisProgram(cfg)
    program.run()


if __name__ == "__main__":
    main()
