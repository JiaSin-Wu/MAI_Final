from _common import *

log = logging.getLogger(__name__)

# Converted ImageEncoder checkpoints directory
CHECKPOINT_DIR = Path("/data1/jeffreytsai/MAI")
MODELS = ["ViT-L-14"]  # Only using ViT-L-14
DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def pretrained_model_path(model_name: str) -> Path:
    """
    Path for the pretrained model (zeroshot).
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    path = CHECKPOINT_DIR / model_name / "zeroshot.pt"
    assert path.is_file(), f"Pretrained model not found at {path}"
    return path


def finetuned_model_path(model_name: str, dataset_name: str) -> Path:
    """
    Path for converted finetuned models (ImageEncoder format).
    Format: /data1/jeffreytsai/MAI/ViT-L-14/{dataset}/finetuned.pt
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")

    path = CHECKPOINT_DIR / model_name / dataset_name / "finetuned.pt"
    assert path.is_file(), f"Finetuned model not found at {path}"
    return path


def main():
    print(f"Checking ViT-L-14...")
    try:
        pretrained_model_path("ViT-L-14")
        print(f"  ✓ Pretrained model found")
    except AssertionError as e:
        print(f"  ✗ {e}")

    for dataset in DATASETS:
        try:
            path = finetuned_model_path("ViT-L-14", dataset)
            print(f"  ✓ {dataset}: {path}")
        except AssertionError as e:
            print(f"  ✗ {dataset}: not found")


if __name__ == "__main__":
    main()

__all__ = [n for n in globals().keys() if not n.startswith("_") and n not in ["log", "main"]]
