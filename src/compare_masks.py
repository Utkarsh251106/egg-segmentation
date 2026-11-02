
import cv2
import numpy as np

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

SAM_DIR = ROOT / "outputs" / "sam_masks" / "train"

GRABCUT_DIR = ROOT / "outputs" / "grabcut_validation" / "masks"

OUTPUT_DIR = ROOT / "outputs" / "mask_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inspect the worst-performing image first
IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)


# =====================================================
# LOAD FILES
# =====================================================

image_path = IMAGE_DIR / IMAGE_NAME

sam_path = SAM_DIR / f"{Path(IMAGE_NAME).stem}_combined.png"

grabcut_path = GRABCUT_DIR / f"{Path(IMAGE_NAME).stem}.png"

image = cv2.imread(str(image_path))

sam_mask = cv2.imread(
    str(sam_path),
    cv2.IMREAD_GRAYSCALE
)

grabcut_mask = cv2.imread(
    str(grabcut_path),
    cv2.IMREAD_GRAYSCALE
)

if image is None or sam_mask is None or grabcut_mask is None:
    raise FileNotFoundError("One or more input files are missing.")

# Ensure identical dimensions
height, width = image.shape[:2]

sam_mask = cv2.resize(
    sam_mask,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

grabcut_mask = cv2.resize(
    grabcut_mask,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

sam = sam_mask > 127
grabcut = grabcut_mask > 127


# =====================================================
# CREATE DISAGREEMENT MAP
# =====================================================

comparison = np.zeros_like(image)

# Green: Both identify foreground
comparison[np.logical_and(sam, grabcut)] = (0, 255, 0)

# Red: GrabCut only
comparison[np.logical_and(~sam, grabcut)] = (0, 0, 255)

# Blue: SAM only
comparison[np.logical_and(sam, ~grabcut)] = (255, 0, 0)


# =====================================================
# DRAW MASK CONTOURS
# =====================================================

sam_contours, _ = cv2.findContours(
    sam_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

grabcut_contours, _ = cv2.findContours(
    grabcut_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

sam_overlay = image.copy()
grabcut_overlay = image.copy()

cv2.drawContours(
    sam_overlay,
    sam_contours,
    -1,
    (255, 0, 0),
    2
)

cv2.drawContours(
    grabcut_overlay,
    grabcut_contours,
    -1,
    (0, 0, 255),
    2
)


# =====================================================
# SAVE COMPARISON
# =====================================================

cv2.imwrite(
    str(OUTPUT_DIR / "original.jpg"),
    image
)

cv2.imwrite(
    str(OUTPUT_DIR / "sam_reference.jpg"),
    sam_overlay
)

cv2.imwrite(
    str(OUTPUT_DIR / "grabcut_prediction.jpg"),
    grabcut_overlay
)

cv2.imwrite(
    str(OUTPUT_DIR / "disagreement.jpg"),
    comparison
)

print("Comparison images saved successfully!")

print("Green = Agreement")
print("Red = GrabCut only")
print("Blue = SAM only")
