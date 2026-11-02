
from pathlib import Path

import cv2
import numpy as np
import torch

from model import build_model


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "test" / "egg"
CHECKPOINT_PATH = ROOT / "models" / "best_egg_unet.pth"

OUTPUT_DIR = ROOT / "outputs" / "test_predictions"
MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_mask(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    resized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    mask = (probabilities.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Return prediction at original image dimensions
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return image, mask


def main():
    print(f"Using device: {DEVICE}")

    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    )
    model.eval()

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    print(f"Test images: {len(image_paths)}")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"Could not read: {image_path.name}")
            continue

        image_rgb, mask = predict_mask(model, image_bgr)

        # Save binary mask: 0 background, 255 egg
        mask_path = MASK_DIR / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask * 255)

        # Red overlay
        overlay = image_rgb.copy()
        foreground = mask > 0

        overlay[foreground] = (
            0.55 * overlay[foreground]
            + 0.45 * np.array([255, 40, 40])
        ).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        overlay_path = OVERLAY_DIR / f"{image_path.stem}.jpg"
        cv2.imwrite(str(overlay_path), overlay_bgr)

        print(f"Saved prediction: {image_path.name}")

    print(f"\nMasks: {MASK_DIR}")
    print(f"Overlays: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
