
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"
MASK_DIR = ROOT / "outputs" / "batch_sam_masks" / "train" / "egg"
OUTPUT_PATH = ROOT / "outputs" / "annotation_review.jpg"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
)

# Contact sheet settings
thumb_w, thumb_h = 240, 180
label_h = 28
cols = 3
rows = math.ceil(len(images) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for idx, image_path in enumerate(images):
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping: {image_path.name}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize mask to image dimensions if required
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    binary_mask = mask > 0

    # Apply red overlay to foreground
    overlay = image.copy()
    overlay[binary_mask] = (
        0.55 * overlay[binary_mask]
        + 0.45 * np.array([255, 40, 40])
    ).astype(np.uint8)

    # Draw mask boundary in yellow
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 255), 2)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(overlay)
    pil_image.thumbnail((thumb_w, thumb_h))

    x = (idx % cols) * thumb_w
    y = (idx // cols) * (thumb_h + label_h)

    sheet.paste(pil_image, (x, y + label_h))
    draw.text(
        (x + 4, y + 4),
        image_path.name[:35],
        fill="black"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_PATH, quality=90)

print(f"Contact sheet saved to: {OUTPUT_PATH}")
