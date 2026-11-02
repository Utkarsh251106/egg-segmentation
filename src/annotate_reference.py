
import cv2
import numpy as np

from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"

OUTPUT_DIR = ROOT / "outputs" / "ground_truth" / "train"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)

points = []

display_image = None
original_image = None


# =====================================================
# MOUSE CALLBACK
# =====================================================

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print("Point added:", x, y)


# =====================================================
# DRAW ANNOTATION
# =====================================================

def draw_annotation():

    canvas = original_image.copy()

    # Draw polygon lines
    for i in range(1, len(points)):

        cv2.line(
            canvas,
            points[i - 1],
            points[i],
            (0, 255, 0),
            2
        )

    # Draw polygon points
    for point in points:

        cv2.circle(
            canvas,
            point,
            4,
            (0, 0, 255),
            -1
        )

    # Close preview polygon
    if len(points) >= 3:

        cv2.line(
            canvas,
            points[-1],
            points[0],
            (255, 255, 0),
            1
        )

    return canvas


# =====================================================
# CREATE BINARY MASK
# =====================================================

def create_mask():

    height, width = original_image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    polygon = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    return mask


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    image_path = IMAGE_DIR / IMAGE_NAME

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:

        raise FileNotFoundError(image_path)

    window = "Egg Ground Truth Annotation"

    cv2.namedWindow(window)

    cv2.setMouseCallback(
        window,
        mouse_callback
    )

    print("\n========== ANNOTATION INSTRUCTIONS ==========")
    print("LEFT CLICK : Add polygon point")
    print("U          : Undo last point")
    print("R          : Reset annotation")
    print("S          : Save binary mask")
    print("Q          : Quit without saving")

    while True:

        display_image = draw_annotation()

        cv2.imshow(
            window,
            display_image
        )

        key = cv2.waitKey(20) & 0xFF

        # Undo
        if key == ord("u"):

            if points:
                points.pop()

        # Reset
        elif key == ord("r"):

            points.clear()

            print("Annotation reset.")

        # Save
        elif key == ord("s"):

            if len(points) < 3:

                print("At least 3 points are required.")
                continue

            mask = create_mask()

            output_path = (
                OUTPUT_DIR /
                f"{Path(IMAGE_NAME).stem}.png"
            )

            cv2.imwrite(
                str(output_path),
                mask
            )

            print("\nGround truth mask saved:")
            print(output_path)

            # Save overlay for verification
            overlay = original_image.copy()

            green = np.zeros_like(original_image)
            green[:] = (0, 255, 0)

            colored = cv2.bitwise_and(
                green,
                green,
                mask=mask
            )

            overlay = cv2.addWeighted(
                original_image,
                0.65,
                colored,
                0.35,
                0
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR /
                    f"{Path(IMAGE_NAME).stem}_overlay.jpg"
                ),
                overlay
            )

            break

        # Quit
        elif key == ord("q"):

            print("Annotation cancelled.")
            break

    cv2.destroyAllWindows()
