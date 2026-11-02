
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH_DIR = ROOT / "outputs" / "ground_truth" / "train"

PREDICTION_DIR = ROOT / "outputs" / "sam_prompt_masks"

OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "sam_prompt_evaluation.csv"


# =====================================================
# METRICS
# =====================================================

def calculate_metrics(gt, pred):

    gt = gt > 0
    pred = pred > 0

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

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

    return {
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


# =====================================================
# EVALUATION
# =====================================================

def main():

    results = []

    gt_files = list(GROUND_TRUTH_DIR.glob("*.png"))

    if not gt_files:

        raise FileNotFoundError(
            f"No ground truth masks found in {GROUND_TRUTH_DIR}"
        )

    for gt_path in gt_files:

        pred_path = PREDICTION_DIR / gt_path.name

        if not pred_path.exists():

            print("Prediction missing:", gt_path.name)

            continue

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        pred = cv2.imread(
            str(pred_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gt is None or pred is None:

            print("Unable to read:", gt_path.name)

            continue

        if gt.shape != pred.shape:

            print(
                f"Shape mismatch: {gt_path.name}",
                gt.shape,
                pred.shape
            )

            continue

        # Convert to binary masks
        gt = (gt > 127).astype(np.uint8)

        pred = (pred > 127).astype(np.uint8)

        metrics = calculate_metrics(gt, pred)

        metrics["image"] = gt_path.name

        results.append(metrics)

        print("\nImage:", gt_path.name)

        for key, value in metrics.items():

            if key != "image":

                print(f"{key}: {value}")

    if not results:

        print("No images were evaluated.")

        return

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_CSV,
        index=False
    )

    print("\n========== FINAL EVALUATION ==========")

    print(df.to_string(index=False))

    print("\nMean IoU:", round(df["IoU"].mean(), 4))

    print("Mean Dice:", round(df["Dice"].mean(), 4))

    print("\nSaved:", RESULTS_CSV)


if __name__ == "__main__":

    main()
