
import cv2
import numpy as np

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

INSPECTION_DIR = ROOT / "outputs" / "sam_inspection"

OUTPUT_PATH = INSPECTION_DIR / "sam_contact_sheet.jpg"

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)

THUMB_WIDTH = 280
THUMB_HEIGHT = 280

COLUMNS = 3


# =====================================================
# LOAD ORIGINAL IMAGE
# =====================================================

image = cv2.imread(
    str(IMAGE_DIR / IMAGE_NAME)
)

if image is None:
    raise FileNotFoundError("Original image not found")


# =====================================================
# LOAD MASK OVERLAYS
# =====================================================

mask_paths = sorted(
    INSPECTION_DIR.glob("mask_*_overlay.jpg"),
    key=lambda p: int(p.stem.split("_")[1])
)

if not mask_paths:
    raise FileNotFoundError("No mask overlays found")


# =====================================================
# CREATE THUMBNAILS
# =====================================================

thumbnails = []

for index, path in enumerate(mask_paths, start=1):

    img = cv2.imread(str(path))

    if img is None:
        continue

    img = cv2.resize(
        img,
        (THUMB_WIDTH, THUMB_HEIGHT)
    )

    # Add mask number
    cv2.putText(
        img,
        f"SAM MASK {index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    thumbnails.append(img)


# =====================================================
# CREATE CONTACT SHEET
# =====================================================

rows = []

for i in range(0, len(thumbnails), COLUMNS):

    row_images = thumbnails[i:i + COLUMNS]

    while len(row_images) < COLUMNS:

        row_images.append(
            np.zeros(
                (THUMB_HEIGHT, THUMB_WIDTH, 3),
                dtype=np.uint8
            )
        )

    rows.append(
        np.hstack(row_images)
    )


contact_sheet = np.vstack(rows)

cv2.imwrite(
    str(OUTPUT_PATH),
    contact_sheet
)

print("Contact sheet created!")
print("Total masks:", len(thumbnails))
print("Saved:", OUTPUT_PATH)
