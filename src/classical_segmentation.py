
import cv2
import numpy as np
import pandas as pd
import time

from pathlib import Path


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

OUTPUT_ROOT = ROOT / "outputs" / "classical_masks"

IMAGE_NAME = (
    "not_damaged_100_jpg.rf."
    "5d780a6024004fa2f68741fa56ddf4c9.jpg"
)

ITERATIONS = 5

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. COMMON MASK REFINEMENT
# =====================================================

def refine_mask(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = np.zeros_like(mask)

    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 500
    ]

    if valid_contours:

        largest = max(
            valid_contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            result,
            [largest],
            -1,
            255,
            cv2.FILLED
        )

    return result


# =====================================================
# 3. E1: GRAYSCALE + OTSU
# =====================================================

def segment_otsu(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Determine foreground polarity from image borders
    border = np.concatenate([
        threshold[0, :],
        threshold[-1, :],
        threshold[:, 0],
        threshold[:, -1]
    ])

    if np.mean(border) > 127:

        threshold = cv2.bitwise_not(
            threshold
        )

    return refine_mask(threshold)


# =====================================================
# 4. E2: RECTANGLE GRABCUT
# =====================================================

def segment_grabcut_rectangle(image):

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    rect = (
        int(width * 0.05),
        int(height * 0.05),
        int(width * 0.90),
        int(height * 0.90)
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        ITERATIONS,
        cv2.GC_INIT_WITH_RECT
    )

    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return refine_mask(result)


# =====================================================
# 5. E3: SEEDED GRABCUT
# =====================================================

def segment_seeded_grabcut(image):

    height, width = image.shape[:2]

    mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    center = (
        width // 2,
        height // 2
    )

    # Probable foreground
    cv2.ellipse(
        mask,
        center,
        (
            int(width * 0.31),
            int(height * 0.31)
        ),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1
    )

    # Definite foreground
    cv2.ellipse(
        mask,
        center,
        (
            int(width * 0.18),
            int(height * 0.18)
        ),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1
    )

    border = max(
        5,
        int(min(width, height) * 0.02)
    )

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        None,
        bgd_model,
        fgd_model,
        ITERATIONS,
        cv2.GC_INIT_WITH_MASK
    )

    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return refine_mask(result)


# =====================================================
# 6. E4: LAB + CLAHE + ADAPTIVE THRESHOLDING
# =====================================================

def segment_lab_adaptive(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        lightness
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        7
    )

    return refine_mask(threshold)


# =====================================================
# 7. E5: OTSU-GUIDED GRABCUT
# =====================================================

def segment_otsu_guided_grabcut(image):

    height, width = image.shape[:2]

    # Initial segmentation using Otsu
    initial_mask = segment_otsu(image)

    points = cv2.findNonZero(
        initial_mask
    )

    if points is None:

        return np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Calculate bounding box
    x, y, w, h = cv2.boundingRect(points)

    # Expand bounding box
    margin_x = int(w * 0.30)
    margin_y = int(h * 0.30)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)

    rect = (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )

    if rect[2] < 10 or rect[3] < 10:

        return initial_mask

    # Initialize GrabCut
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        ITERATIONS,
        cv2.GC_INIT_WITH_RECT
    )

    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return refine_mask(result)


# =====================================================
# 8. VISUALIZATION
# =====================================================

def create_overlay(image, mask):

    overlay = image.copy()

    green = np.zeros_like(image)

    green[:] = (0, 255, 0)

    segmented = cv2.bitwise_and(
        green,
        green,
        mask=mask
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        segmented,
        0.35,
        0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    return overlay


# =====================================================
# 9. SAVE RESULTS
# =====================================================

def save_result(image, mask, experiment_name):

    output_dir = OUTPUT_ROOT / experiment_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay = create_overlay(
        image,
        mask
    )

    cv2.imwrite(
        str(output_dir / "mask.png"),
        mask
    )

    cv2.imwrite(
        str(output_dir / "overlay.jpg"),
        overlay
    )

    foreground_pixels = int(
        np.count_nonzero(mask)
    )

    area_percentage = (
        foreground_pixels / mask.size
    ) * 100

    return {
        "experiment": experiment_name,
        "foreground_pixels": foreground_pixels,
        "area_percentage": round(area_percentage, 2)
    }


# =====================================================
# 10. MAIN EXECUTION
# =====================================================

def main():

    image_path = IMAGE_DIR / IMAGE_NAME

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            "Unable to read input image."
        )

    print("\nProcessing:", image_path.name)

    experiments = {
        "experiment_1_otsu": segment_otsu,
        "experiment_2_grabcut": segment_grabcut_rectangle,
        "experiment_3_seeded_grabcut": segment_seeded_grabcut,
        "experiment_4_lab_adaptive": segment_lab_adaptive,
        "experiment_5_otsu_guided_grabcut": segment_otsu_guided_grabcut
    }

    results = []

    for name, algorithm in experiments.items():

        print("\nRunning:", name)

        start_time = time.time()

        mask = algorithm(image)

        elapsed = time.time() - start_time

        result = save_result(
            image,
            mask,
            name
        )

        result["execution_time"] = round(
            elapsed,
            3
        )

        results.append(result)

        print(
            "Foreground pixels:",
            result["foreground_pixels"]
        )

        print(
            "Area percentage:",
            result["area_percentage"],
            "%"
        )

        print(
            "Execution time:",
            result["execution_time"],
            "seconds"
        )

    # Save experiment comparison
    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_ROOT / "experiment_comparison.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n========== EXPERIMENT SUMMARY ==========")

    print(results_df.to_string(index=False))

    print("\nCSV saved:", csv_path)

    print("\nAll five experiments completed!")


# =====================================================
# 11. ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
