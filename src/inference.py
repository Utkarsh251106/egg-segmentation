
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch

from model import build_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "best_egg_unet.pth"
IMAGE_SIZE = 256
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained U-Net model."""
    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


def preprocess_image(image):
    """Resize and normalize an image for model input."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # ImageNet normalization used by the ResNet34 encoder.
    resized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized = (resized - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
    tensor = tensor.unsqueeze(0).to(DEVICE)

    return tensor


@torch.inference_mode()
def predict(model, image):
    """Generate a binary egg mask at the original image resolution."""
    input_tensor = preprocess_image(image)

    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits)

    mask = (probabilities.squeeze().cpu().numpy() >= THRESHOLD).astype(
        np.uint8
    )

    # Restore the predicted mask to the original image dimensions.
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask * 255


def create_overlay(image, mask):
    """Overlay the predicted egg region in red."""
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)

    result = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

    # Draw the mask boundary.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Egg segmentation using trained U-Net"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input egg image",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference",
        help="Folder to save the predicted mask and overlay",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise SystemExit(f"Model checkpoint not found: {CHECKPOINT}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    print(f"Loading model on {DEVICE}...")
    model = load_model()

    print("Running prediction...")
    mask = predict(model, image)
    overlay = create_overlay(image, mask)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.jpg"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    print("\nInference complete.")
    print(f"Mask saved:    {mask_path}")
    print(f"Overlay saved: {overlay_path}")


if __name__ == "__main__":
    main()
