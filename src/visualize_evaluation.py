
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "test" / "egg"
GT_DIR = ROOT / "outputs" / "test_predictions" / "ground_truth"
PRED_DIR = ROOT / "outputs" / "test_predictions" / "masks"
OUTPUT_DIR = ROOT / "outputs" / "test_predictions" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for gt_path in sorted(GT_DIR.glob("*.png")):
    image_path = next(
        (
            p for p in IMAGE_DIR.iterdir()
            if p.stem == gt_path.stem
        ),
        None,
    )

    pred_path = PRED_DIR / gt_path.name

    if image_path is None or not pred_path.exists():
        print(f"Skipping: {gt_path.stem}")
        continue

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) > 127

    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Green = correct egg pixels, red = false positives,
    # blue = false negatives.
    comparison = np.zeros_like(image)
    comparison[np.logical_and(pred, gt)] = [0, 255, 0]
    comparison[np.logical_and(pred, ~gt)] = [255, 0, 0]
    comparison[np.logical_and(~pred, gt)] = [0, 0, 255]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("U-Net Prediction")

    axes[3].imshow(comparison)
    axes[3].set_title("Comparison")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{gt_path.stem}_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nVisual evaluation complete.")
