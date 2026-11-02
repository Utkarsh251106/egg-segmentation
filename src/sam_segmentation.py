
import cv2
import torch
import numpy as np
import pandas as pd
import time

from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"

DATASET_DIR = ROOT / "dataset"

OUTPUT_ROOT = ROOT / "outputs" / "sam_masks"

DEVICE = "cpu"

SPLITS = ["train", "valid", "test"]

# Mask filtering parameters
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.80

MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20

DUPLICATE_IOU_THRESHOLD = 0.90


# =====================================================
# 2. DIRECTORY SETUP
# =====================================================

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"SAM checkpoint not found: {CHECKPOINT}"
    )

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

if OUTPUT_ROOT.exists() and not OUTPUT_ROOT.is_dir():
    raise NotADirectoryError(
        f"{OUTPUT_ROOT} exists as a file."
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =====================================================
# 3. LOAD SAM MODEL
# =====================================================

def load_sam():

    print("\nLoading SAM ViT-B model...")

    model = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT)
    )

    model.to(device=DEVICE)
    model.eval()

    print("SAM loaded successfully!")
    print("Device:", DEVICE)

    return model


sam = load_sam()


# =====================================================
# 4. AUTOMATIC MASK GENERATOR
# =====================================================

mask_generator = SamAutomaticMaskGenerator(
    model=sam,

    points_per_side=16,

    pred_iou_thresh=0.86,

    stability_score_thresh=0.92,

    crop_n_layers=0,

    min_mask_region_area=100
)


# =====================================================
# 5. FILTER MASKS
# =====================================================

def filter_masks(masks, image_area):

    filtered = []

    for mask_data in masks:

        area = mask_data["area"]

        x, y, w, h = mask_data["bbox"]

        area_ratio = area / image_area

        # Remove tiny regions
        if area_ratio < MIN_AREA_RATIO:
            continue

        # Remove nearly full-image regions
        if area_ratio > MAX_AREA_RATIO:
            continue

        # Remove small bounding boxes
        if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
            continue

        filtered.append(mask_data)

    return filtered


# =====================================================
# 6. REMOVE DUPLICATE MASKS
# =====================================================

def remove_duplicate_masks(masks):

    masks = sorted(
        masks,
        key=lambda x: x["predicted_iou"],
        reverse=True
    )

    selected = []

    for candidate in masks:

        candidate_mask = candidate["segmentation"]

        is_duplicate = False

        for existing in selected:

            existing_mask = existing["segmentation"]

            intersection = np.logical_and(
                candidate_mask,
                existing_mask
            ).sum()

            union = np.logical_or(
                candidate_mask,
                existing_mask
            ).sum()

            iou = intersection / union if union else 0

            if iou > DUPLICATE_IOU_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(candidate)

    return selected


# =====================================================
# 7. CREATE VISUALIZATION
# =====================================================

def create_overlay(image, masks):

    overlay = image.copy()

    rng = np.random.default_rng(42)

    for mask_data in masks:

        mask = mask_data["segmentation"]

        color = rng.integers(
            50, 256, size=3
        )

        overlay[mask] = (
            0.5 * overlay[mask] +
            0.5 * color
        ).astype(np.uint8)

    return overlay


# =====================================================
# 8. PROCESS IMAGE
# =====================================================

def process_image(image_path, output_dir, binary_dir):

    print("\nProcessing:", image_path.name)

    image = cv2.imread(str(image_path))

    if image is None:

        print("Image reading failed. Skipping.")

        return None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]

    image_area = height * width

    start_time = time.time()

    # Generate masks
    with torch.inference_mode():

        masks = mask_generator.generate(image_rgb)

    original_count = len(masks)

    # Filter masks
    filtered = filter_masks(
        masks,
        image_area
    )

    filtered_count = len(filtered)

    # Remove duplicates
    final_masks = remove_duplicate_masks(
        filtered
    )

    final_count = len(final_masks)

    # Create overlay
    overlay = create_overlay(
        image_rgb,
        final_masks
    )

    output_name = image_path.stem

    # Save overlay
    overlay_path = output_dir / (
        f"{output_name}_overlay.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # Combined binary mask
    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for index, mask_data in enumerate(final_masks):

        mask = mask_data["segmentation"]

        combined_mask[mask] = 255

        # Save individual mask
        individual_mask = (
            mask.astype(np.uint8) * 255
        )

        mask_path = binary_dir / (
            f"{output_name}_mask_{index + 1}.png"
        )

        cv2.imwrite(
            str(mask_path),
            individual_mask
        )

    # Save combined mask
    combined_path = output_dir / (
        f"{output_name}_combined.png"
    )

    cv2.imwrite(
        str(combined_path),
        combined_mask
    )

    elapsed = time.time() - start_time

    print("Original masks:", original_count)
    print("After filtering:", filtered_count)
    print("Final masks:", final_count)
    print("Execution time:", round(elapsed, 2), "seconds")

    return {
        "image": image_path.name,
        "original_masks": original_count,
        "filtered_masks": filtered_count,
        "final_masks": final_count,
        "execution_time": round(elapsed, 2)
    }


# =====================================================
# 9. PROCESS COMPLETE DATASET
# =====================================================

def main():

    all_results = []

    total_images = 0

    for split in SPLITS:

        image_dir = (
            DATASET_DIR / split / "egg"
        )

        if not image_dir.is_dir():

            print(f"\nSkipping {split}: folder not found")

            continue

        # Support common image extensions
        image_paths = []

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png"
        ]:

            image_paths.extend(
                image_dir.glob(extension)
            )

        image_paths = sorted(image_paths)

        print("\n" + "=" * 50)
        print(f"PROCESSING {split.upper()}")
        print("=" * 50)

        print("Images found:", len(image_paths))

        total_images += len(image_paths)

        # Split-specific output directories
        output_dir = OUTPUT_ROOT / split

        binary_dir = output_dir / "binary_masks"

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        binary_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, image_path in enumerate(image_paths):

            print(
                f"\nImage {index + 1}/{len(image_paths)}"
            )

            try:

                result = process_image(
                    image_path,
                    output_dir,
                    binary_dir
                )

                if result:

                    result["split"] = split

                    all_results.append(result)

            except Exception as error:

                print(
                    f"Error processing {image_path.name}: {error}"
                )

    # =================================================
    # 10. SAVE EXECUTION REPORT
    # =================================================

    results_df = pd.DataFrame(all_results)

    csv_path = OUTPUT_ROOT / "sam_results.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)

    print("Total images found:", total_images)

    print("Successfully processed:", len(results_df))

    if not results_df.empty:

        print("\nSplit-wise image counts:")

        print(
            results_df.groupby("split")["image"].count()
        )

        print(
            "\nAverage inference time:",
            round(
                results_df["execution_time"].mean(),
                2
            ),
            "seconds"
        )

        print(
            "\nTotal execution time:",
            round(
                results_df["execution_time"].sum() / 60,
                2
            ),
            "minutes"
        )

    print("\nCSV saved:", csv_path)

    print("\nSAM dataset processing completed!")


# =====================================================
# 11. ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
