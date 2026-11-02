
import cv2
import numpy as np
from pathlib import Path


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6"
)

IMAGE_PATH = (
    ROOT / "dataset" / "train" / "egg"
    / f"{IMAGE_NAME}.jpg"
)

GT_PATH = (
    ROOT / "outputs" / "ground_truth" / "train"
    / f"{IMAGE_NAME}.png"
)

PRED_PATH = (
    ROOT / "outputs" / "sam_prompt_masks"
    / f"{IMAGE_NAME}.png"
)

OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. LOAD FILES
# =====================================================

image = cv2.imread(str(IMAGE_PATH))

gt_original = cv2.imread(
    str(GT_PATH),
    cv2.IMREAD_GRAYSCALE
)

pred_original = cv2.imread(
    str(PRED_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Original image not found: {IMAGE_PATH}"
    )

if gt_original is None:
    raise FileNotFoundError(
        f"Ground truth mask not found: {GT_PATH}"
    )

if pred_original is None:
    raise FileNotFoundError(
        f"SAM prediction not found: {PRED_PATH}"
    )


# =====================================================
# 3. INSPECT ORIGINAL MASK VALUES
# =====================================================

print("\n========== ORIGINAL MASK DIAGNOSTICS ==========")

print("Image shape:", image.shape)

print("GT shape:", gt_original.shape)

print("Prediction shape:", pred_original.shape)

print("\nGround Truth Pixel Values:")

gt_values, gt_counts = np.unique(
    gt_original,
    return_counts=True
)

for value, count in zip(gt_values, gt_counts):

    print(
        f"Value {value}: {count} pixels "
        f"({count / gt_original.size * 100:.2f}%)"
    )

print("\nGround Truth Minimum:", gt_original.min())

print("Ground Truth Maximum:", gt_original.max())

print("\nForeground at different thresholds:")

for threshold in [0, 1, 10, 127, 254]:

    count = np.count_nonzero(
        gt_original > threshold
    )

    percentage = (
        count / gt_original.size
    ) * 100

    print(
        f"Threshold {threshold}: "
        f"{count} pixels ({percentage:.2f}%)"
    )


# =====================================================
# 4. PREPARE BINARY MASKS
# =====================================================

# Do not resize masks automatically.
# A shape mismatch should be investigated.

if gt_original.shape != pred_original.shape:

    raise ValueError(
        "\nGround truth and prediction dimensions differ.\n"
        f"GT: {gt_original.shape}\n"
        f"Prediction: {pred_original.shape}\n"
        "Check whether masks belong to the same image."
    )

if gt_original.shape != image.shape[:2]:

    raise ValueError(
        "\nGround truth and original image dimensions differ.\n"
        f"GT: {gt_original.shape}\n"
        f"Image: {image.shape[:2]}"
    )


# Convert to binary masks

gt = (
    gt_original > 127
).astype(np.uint8)

pred = (
    pred_original > 127
).astype(np.uint8)


# =====================================================
# 5. FOREGROUND STATISTICS
# =====================================================

gt_pixels = np.count_nonzero(gt)

pred_pixels = np.count_nonzero(pred)

total_pixels = gt.size

print("\n========== BINARY MASK STATISTICS ==========")

print("Total pixels:", total_pixels)

print("GT foreground pixels:", gt_pixels)

print("Prediction foreground pixels:", pred_pixels)

print(
    "GT foreground percentage:",
    round(gt_pixels / total_pixels * 100, 2),
    "%"
)

print(
    "Prediction foreground percentage:",
    round(pred_pixels / total_pixels * 100, 2),
    "%"
)


# =====================================================
# 6. CHECK EMPTY OR ABNORMAL GROUND TRUTH
# =====================================================

if gt_pixels == 0:

    raise ValueError(
        "Ground truth is completely empty. "
        "The annotation mask must be corrected."
    )

if gt_pixels < 1000:

    print("\nWARNING:")
    print(
        "Ground truth contains fewer than 1000 foreground pixels."
    )

    print(
        "This is suspicious for the egg in this image."
    )

    print(
        "Inspect the saved ground-truth mask before trusting metrics."
    )


# =====================================================
# 7. CALCULATE METRICS
# =====================================================

tp = np.logical_and(
    gt == 1,
    pred == 1
).sum()

fp = np.logical_and(
    gt == 0,
    pred == 1
).sum()

fn = np.logical_and(
    gt == 1,
    pred == 0
).sum()

tn = np.logical_and(
    gt == 0,
    pred == 0
).sum()

epsilon = 1e-7

iou = tp / (tp + fp + fn + epsilon)

dice = (
    2 * tp
    / (2 * tp + fp + fn + epsilon)
)

precision = tp / (tp + fp + epsilon)

recall = tp / (tp + fn + epsilon)

f1 = (
    2 * precision * recall
    / (precision + recall + epsilon)
)

accuracy = (
    (tp + tn)
    / (tp + tn + fp + fn + epsilon)
)


# =====================================================
# 8. PRINT METRICS
# =====================================================

print("\n========== SEGMENTATION METRICS ==========")

print("TP:", int(tp))

print("FP:", int(fp))

print("FN:", int(fn))

print("TN:", int(tn))

print("IoU:", round(float(iou), 4))

print("Dice:", round(float(dice), 4))

print("Precision:", round(float(precision), 4))

print("Recall:", round(float(recall), 4))

print("F1:", round(float(f1), 4))

print("Accuracy:", round(float(accuracy), 4))


# =====================================================
# 9. CREATE COMPARISON IMAGE
# =====================================================

comparison = np.zeros_like(image)

gt_only = np.logical_and(
    gt == 1,
    pred == 0
)

pred_only = np.logical_and(
    gt == 0,
    pred == 1
)

agreement = np.logical_and(
    gt == 1,
    pred == 1
)

# Green = GT only
comparison[gt_only] = (0, 255, 0)

# Red = Prediction only
comparison[pred_only] = (0, 0, 255)

# Yellow = Agreement
comparison[agreement] = (0, 255, 255)


# =====================================================
# 10. CREATE OVERLAY
# =====================================================

overlay = image.copy()

# Ground truth boundary: green

gt_contours, _ = cv2.findContours(
    gt,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    overlay,
    gt_contours,
    -1,
    (0, 255, 0),
    2
)


# Prediction boundary: red

pred_contours, _ = cv2.findContours(
    pred,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    overlay,
    pred_contours,
    -1,
    (0, 0, 255),
    2
)


# =====================================================
# 11. SAVE OUTPUTS
# =====================================================

comparison_path = (
    OUTPUT_DIR / "mask_comparison.png"
)

overlay_path = (
    OUTPUT_DIR / "mask_overlay.png"
)

gt_visual_path = (
    OUTPUT_DIR / "ground_truth_visualization.png"
)

pred_visual_path = (
    OUTPUT_DIR / "prediction_visualization.png"
)


cv2.imwrite(
    str(comparison_path),
    comparison
)

cv2.imwrite(
    str(overlay_path),
    overlay
)

cv2.imwrite(
    str(gt_visual_path),
    gt * 255
)

cv2.imwrite(
    str(pred_visual_path),
    pred * 255
)


# =====================================================
# 12. FINAL REPORT
# =====================================================

print("\n========== SAVED OUTPUTS ==========")

print("Comparison:", comparison_path)

print("Overlay:", overlay_path)

print("Ground Truth:", gt_visual_path)

print("Prediction:", pred_visual_path)

print("\nInspection completed!")
