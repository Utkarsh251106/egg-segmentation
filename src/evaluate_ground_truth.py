
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)

STEM = Path(IMAGE_NAME).stem

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

GT_DIR = ROOT / "outputs" / "ground_truth" / "train"

PRED_DIR = ROOT / "outputs" / "grabcut_validation" / "masks"

OUTPUT_DIR = ROOT / "outputs" / "ground_truth_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD INPUT FILES
# =====================================================

image = cv2.imread(
    str(IMAGE_DIR / IMAGE_NAME)
)

gt = cv2.imread(
    str(GT_DIR / f"{STEM}.png"),
    cv2.IMREAD_GRAYSCALE
)

prediction = cv2.imread(
    str(PRED_DIR / f"{STEM}.png"),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Original image missing")

if gt is None:
    raise FileNotFoundError("Ground truth mask missing")

if prediction is None:
    raise FileNotFoundError("GrabCut prediction missing")


# =====================================================
# 3. PREPARE BINARY MASKS
# =====================================================

height, width = image.shape[:2]

gt = cv2.resize(
    gt,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

prediction = cv2.resize(
    prediction,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

gt = gt > 127
prediction = prediction > 127


# =====================================================
# 4. CALCULATE METRICS
# =====================================================

tp = np.logical_and(gt, prediction).sum()

fp = np.logical_and(~gt, prediction).sum()

fn = np.logical_and(gt, ~prediction).sum()

tn = np.logical_and(~gt, ~prediction).sum()

iou = tp / (tp + fp + fn) if tp + fp + fn else 0

dice = (
    2 * tp / (2 * tp + fp + fn)
    if 2 * tp + fp + fn else 0
)

precision = (
    tp / (tp + fp)
    if tp + fp else 0
)

recall = (
    tp / (tp + fn)
    if tp + fn else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

accuracy = (
    (tp + tn) / (tp + tn + fp + fn)
)


# =====================================================
# 5. CREATE DISAGREEMENT MAP
# =====================================================

comparison = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

# Green: Correct foreground
comparison[gt & prediction] = (0, 255, 0)

# Red: False positive
comparison[~gt & prediction] = (0, 0, 255)

# Blue: False negative
comparison[gt & ~prediction] = (255, 0, 0)


# =====================================================
# 6. CREATE PREDICTION OVERLAY
# =====================================================

overlay = image.copy()

green = np.zeros_like(image)

green[:] = (0, 255, 0)

colored_mask = cv2.bitwise_and(
    green,
    green,
    mask=prediction.astype(np.uint8) * 255
)

overlay = cv2.addWeighted(
    image,
    0.65,
    colored_mask,
    0.35,
    0
)

# Ground truth boundary: blue
gt_uint8 = gt.astype(np.uint8) * 255

contours, _ = cv2.findContours(
    gt_uint8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    overlay,
    contours,
    -1,
    (255, 0, 0),
    2
)


# =====================================================
# 7. SAVE RESULTS
# =====================================================

cv2.imwrite(
    str(OUTPUT_DIR / "disagreement.png"),
    comparison
)

cv2.imwrite(
    str(OUTPUT_DIR / "ground_truth_vs_prediction.jpg"),
    overlay
)

results = {
    "image": IMAGE_NAME,
    "IoU": round(float(iou), 4),
    "Dice": round(float(dice), 4),
    "Precision": round(float(precision), 4),
    "Recall": round(float(recall), 4),
    "F1": round(float(f1), 4),
    "Accuracy": round(float(accuracy), 4),
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "TN": int(tn)
}

df = pd.DataFrame([results])

df.to_csv(
    OUTPUT_DIR / "evaluation_results.csv",
    index=False
)


# =====================================================
# 8. PRINT REPORT
# =====================================================

print("\n========== GROUND TRUTH EVALUATION ==========")

for key, value in results.items():

    print(f"{key}: {value}")

print("\nSaved outputs:")
print(OUTPUT_DIR)

print("\nEvaluation completed!")
