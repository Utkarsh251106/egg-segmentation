
import cv2
import numpy as np

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

MASK_DIR = (
    ROOT / "outputs" / "sam_masks" /
    "train" / "binary_masks"
)

OUTPUT_DIR = ROOT / "outputs" / "sam_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)


# =====================================================
# LOAD ORIGINAL IMAGE
# =====================================================

image_path = IMAGE_DIR / IMAGE_NAME

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)


# =====================================================
# INSPECT INDIVIDUAL SAM MASKS
# =====================================================

stem = image_path.stem

mask_paths = sorted(
    MASK_DIR.glob(f"{stem}_mask_*.png")
)

print("Total individual SAM masks:", len(mask_paths))

if not mask_paths:
    raise FileNotFoundError(
        "No individual SAM masks found."
    )


# =====================================================
# GENERATE MASK VISUALIZATIONS
# =====================================================

summary = []

for index, mask_path in enumerate(mask_paths, start=1):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue

    binary = mask > 127

    area = int(np.count_nonzero(binary))

    percentage = area / binary.size * 100

    # Create overlay
    overlay = image.copy()

    green = np.zeros_like(image)
    green[:] = (0, 255, 0)

    colored_mask = cv2.bitwise_and(
        green,
        green,
        mask=mask
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        colored_mask,
        0.35,
        0
    )

    # Draw boundary
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    output_path = (
        OUTPUT_DIR /
        f"mask_{index}_overlay.jpg"
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    summary.append({
        "mask_id": index,
        "area": area,
        "area_percentage": round(percentage, 2),
        "file": mask_path.name
    })

    print(
        f"Mask {index}: "
        f"Area={area}, "
        f"Percentage={percentage:.2f}%"
    )


# =====================================================
# SAVE SUMMARY
# =====================================================

import pandas as pd

df = pd.DataFrame(summary)

df.to_csv(
    OUTPUT_DIR / "mask_summary.csv",
    index=False
)

print("\nIndividual mask inspection completed!")

print("Results saved:", OUTPUT_DIR)
