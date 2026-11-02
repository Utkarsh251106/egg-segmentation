
import cv2
import numpy as np

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)

IMAGE_PATH = ROOT / "dataset" / "train" / "egg" / IMAGE_NAME

MASK_DIR = ROOT / "outputs" / "ground_truth" / "train"

STEM = Path(IMAGE_NAME).stem

MASK_PATH = MASK_DIR / f"{STEM}.png"

OUTPUT_PATH = MASK_PATH

image = cv2.imread(str(IMAGE_PATH))

mask = cv2.imread(
    str(MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None or mask is None:
    raise FileNotFoundError("Image or ground truth mask missing")


# =====================================================
# STATE
# =====================================================

points = []

mode = "erase"

preview = image.copy()


# =====================================================
# MOUSE CALLBACK
# =====================================================

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print("Point:", x, y)


# =====================================================
# DRAW PREVIEW
# =====================================================

def draw_preview():

    canvas = image.copy()

    # Show existing mask
    green = np.zeros_like(image)
    green[:] = (0, 255, 0)

    colored = cv2.bitwise_and(
        green,
        green,
        mask=mask
    )

    canvas = cv2.addWeighted(
        canvas,
        0.65,
        colored,
        0.35,
        0
    )

    # Draw correction polygon
    for i in range(1, len(points)):

        cv2.line(
            canvas,
            points[i - 1],
            points[i],
            (0, 0, 255),
            2
        )

    for point in points:

        cv2.circle(
            canvas,
            point,
            4,
            (255, 0, 0),
            -1
        )

    cv2.putText(
        canvas,
        f"MODE: {mode.upper()} | ENTER: Apply | U: Undo | R: Reset | S: Save | Q: Quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return canvas


# =====================================================
# MAIN
# =====================================================

window = "Ground Truth Correction"

cv2.namedWindow(window)

cv2.setMouseCallback(
    window,
    mouse_callback
)

print("\n========== CORRECTION TOOL ==========")
print("E       : Erase polygon from mask")
print("F       : Add foreground polygon")
print("ENTER   : Apply polygon")
print("U       : Undo last point")
print("R       : Reset current polygon")
print("S       : Save corrected mask")
print("Q       : Quit")

while True:

    cv2.imshow(
        window,
        draw_preview()
    )

    key = cv2.waitKey(20) & 0xFF

    # Erase mode
    if key == ord("e"):

        mode = "erase"
        points.clear()

    # Foreground mode
    elif key == ord("f"):

        mode = "foreground"
        points.clear()

    # Undo
    elif key == ord("u"):

        if points:
            points.pop()

    # Reset
    elif key == ord("r"):

        points.clear()

    # Apply polygon
    elif key in (13, 10):

        if len(points) < 3:

            print("Add at least 3 points.")
            continue

        polygon = np.array(
            points,
            dtype=np.int32
        )

        if mode == "erase":

            cv2.fillPoly(
                mask,
                [polygon],
                0
            )

        else:

            cv2.fillPoly(
                mask,
                [polygon],
                255
            )

        points.clear()

        print("Polygon applied:", mode)

    # Save
    elif key == ord("s"):

        cv2.imwrite(
            str(OUTPUT_PATH),
            mask
        )

        # Save corrected overlay
        green = np.zeros_like(image)
        green[:] = (0, 255, 0)

        colored = cv2.bitwise_and(
            green,
            green,
            mask=mask
        )

        overlay = cv2.addWeighted(
            image,
            0.65,
            colored,
            0.35,
            0
        )

        cv2.imwrite(
            str(MASK_DIR / f"{STEM}_overlay.jpg"),
            overlay
        )

        print("Corrected mask saved!")

        break

    elif key == ord("q"):

        print("Exited without saving.")
        break

cv2.destroyAllWindows()
