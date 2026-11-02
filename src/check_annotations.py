from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"
MASK_DIR = ROOT / "outputs" / "batch_sam_masks" / "train" / "egg"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
)

print(f"Images found: {len(images)}")

valid_masks = 0
missing_masks = []
empty_masks = []
invalid_masks = []

for image_path in images:
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        missing_masks.append(image_path.name)
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        invalid_masks.append(image_path.name)
        continue

    foreground_pixels = np.count_nonzero(mask)

    if foreground_pixels == 0:
        empty_masks.append(image_path.name)

    valid_masks += 1

    print(
        f"{image_path.name}: "
        f"foreground={foreground_pixels}, "
        f"percentage={(foreground_pixels / mask.size) * 100:.2f}%"
    )

print("\n--- Summary ---")
print(f"Valid masks: {valid_masks}")
print(f"Missing masks: {len(missing_masks)}")
print(f"Empty masks: {len(empty_masks)}")
print(f"Unreadable masks: {len(invalid_masks)}")

if missing_masks:
    print("Missing:", missing_masks)

if empty_masks:
    print("Empty:", empty_masks)

if invalid_masks:
    print("Unreadable:", invalid_masks)