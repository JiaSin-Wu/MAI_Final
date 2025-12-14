import torch
import sys
from pathlib import Path

sys.path.append("/home/jeffreytsai/weight-ensembling_MoE")

from src.modeling import ImageEncoder
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from tqdm import tqdm
from omegaconf import OmegaConf

print("="*80)
print("Testing Distilled Model on Cars Dataset")
print("="*80)

# Create config
cfg = OmegaConf.create({
    "data_location": "/data1/chliu/MAI2025_final/data",
    "model": "ViT-L-14",
    "batch_size": 32,
    "openclip_cachedir": None,
    "cache_dir": None,
    "save": "/data1/chliu/MAI2025_final/cache/checkpoints/task_vectors_checkpoints/ViT-L-14",
    "device": "cuda",
    "num_workers": 1,
})

# Load the distilled Cars model
model_path = Path("/data1/chliu/MAI2025_final/results/clip_distillation/ViT-L-14/checkpoints/distillation_Cars.pt")
print(f"\n1. Loading distilled model from: {model_path}")
assert model_path.exists(), f"Model not found at {model_path}"

state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
print(f"   ✓ State dict loaded")

# Create ImageEncoder model
print(f"\n2. Creating ImageEncoder model")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ImageEncoder(cfg, keep_lang=False)
print(f"   ✓ ImageEncoder created")

# Load the distilled weights
model.load_state_dict(state_dict)
print(f"   ✓ Loaded distilled weights into model")

# Move to GPU
model = model.to(device)
model.eval()
print(f"   ✓ Model moved to {device}")

# Load classification head for Cars
print(f"\n3. Loading classification head for Cars dataset")
classification_head = get_classification_head(cfg, "Cars")
classification_head = classification_head.to(device)
print(f"   ✓ Classification head loaded")

# Load Cars test dataset
print(f"\n4. Loading Cars test dataset")
dataset = get_dataset(
    "Cars",
    model.val_preprocess,
    location=cfg.data_location,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers
)

test_loader = dataset.test_loader
print(f"   ✓ Test loader created")

# Evaluate
print(f"\n5. Evaluating on Cars test dataset")
total_correct = 0
total_count = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        # Get features from image encoder
        features = model(images)

        # Get logits from classification head
        logits = classification_head(features)

        # Get predictions
        preds = logits.argmax(dim=-1)

        # Calculate accuracy
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_count += len(labels)

        # Debug first batch
        if batch_idx == 0:
            print(f"\n   Debug (first batch):")
            print(f"     Features shape: {features.shape}, mean: {features.mean():.4f}, std: {features.std():.4f}")
            print(f"     Logits shape: {logits.shape}, min: {logits.min():.4f}, max: {logits.max():.4f}")
            print(f"     Predictions: {preds[:10].cpu()}")
            print(f"     Labels: {labels[:10].cpu()}")
            print(f"     Unique predictions: {preds.unique().cpu()}")

# Calculate final accuracy
accuracy = total_correct / total_count * 100
print(f"\n{'='*80}")
print(f"Results:")
print(f"{'='*80}")
print(f"Total samples: {total_count}")
print(f"Correct predictions: {total_correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"{'='*80}")

if accuracy < 1.0:
    print(f"\n⚠️  WARNING: Accuracy is very low ({accuracy:.2f}%)")
    print(f"   This suggests the distilled model may not be working correctly")
    print(f"   or the classification head is incompatible")
elif accuracy < 30.0:
    print(f"\n⚠️  Accuracy is lower than expected for Cars dataset")
    print(f"   Expected accuracy should be > 50% for a well-trained model")
else:
    print(f"\n✓ Distilled model appears to be working!")
