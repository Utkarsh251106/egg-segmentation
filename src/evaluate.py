from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "outputs" / "test_predictions" / "masks"
GT_DIR = ROOT / "outputs" / "test_predictions" / "ground_truth"

THRESHOLD = 0.5


def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate segmentation metrics for one image.
    Both masks should be binary arrays.
    """

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Pixel Accuracy": accuracy,
    }


def main():
    if not PRED_DIR.exists():
        raise SystemExit(f"Prediction folder not found: {PRED_DIR}")

    if not GT_DIR.exists():
        raise SystemExit(f"Ground-truth folder not found: {GT_DIR}")

    pred_files = sorted(PRED_DIR.glob("*.png"))

    if not pred_files:
        raise SystemExit(f"No prediction masks found in: {PRED_DIR}")

    results = []

    for pred_path in pred_files:
        gt_path = GT_DIR / pred_path.name

        if not gt_path.exists():
            print(f"Ground-truth mask missing: {pred_path.name}")
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            print(f"Could not read mask: {pred_path.name}")
            continue

        # Ensure prediction and ground truth have matching dimensions.
        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert masks to binary.
        pred_binary = pred >= int(THRESHOLD * 255)
        gt_binary = gt > 127

        metrics = calculate_metrics(pred_binary, gt_binary)
        metrics["Image"] = pred_path.name

        results.append(metrics)

        print(
            f"{pred_path.name} | "
            f"Dice: {metrics['Dice']:.4f} | "
            f"IoU: {metrics['IoU']:.4f}"
        )

    if not results:
        raise SystemExit("No matching prediction and ground-truth masks found.")

    df = pd.DataFrame(results)

    # Per-image average metrics.
    metric_columns = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Pixel Accuracy",
    ]

    print("\n========== TEST RESULTS ==========")
    print(f"Images evaluated: {len(df)}")

    for metric in metric_columns:
        print(f"Mean {metric}: {df[metric].mean():.4f}")

    # Save detailed per-image results.
    output_path = ROOT / "outputs" / "test_predictions" / "evaluation_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()