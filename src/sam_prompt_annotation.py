
import cv2
import numpy as np
import torch

from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = (
    "not_damaged_117_jpg.rf."
    "d25ded9e7408f9b89b3cfb3de581f6a6.jpg"
)

IMAGE_PATH = ROOT / "dataset" / "train" / "egg" / IMAGE_NAME

CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"

OUTPUT_DIR = ROOT / "outputs" / "sam_prompt_masks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = "vit_b"

DEVICE = "cpu"


# =====================================================
# 2. LOAD SAM MODEL
# =====================================================

print("\nLoading SAM ViT-B...")

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(CHECKPOINT)
)

sam.to(device=DEVICE)

predictor = SamPredictor(sam)

print("SAM loaded successfully!")
print("Device:", DEVICE)


# =====================================================
# 3. LOAD IMAGE
# =====================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

predictor.set_image(image_rgb)

height, width = image.shape[:2]


# =====================================================
# 4. PROMPT STATE
# =====================================================

# Label 1 = Positive point (egg)
# Label 0 = Negative point (background)

points = []
labels = []

candidate_masks = None
candidate_scores = None

selected_candidate = 0

window_name = "SAM Egg Annotation"


# =====================================================
# 5. SAM PREDICTION
# =====================================================

def run_prediction():

    global candidate_masks
    global candidate_scores
    global selected_candidate

    if not points:
        candidate_masks = None
        candidate_scores = None
        return

    # At least one positive point is required
    if 1 not in labels:
        candidate_masks = None
        candidate_scores = None
        return

    point_coords = np.array(
        points,
        dtype=np.float32
    )

    point_labels = np.array(
        labels,
        dtype=np.int32
    )

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    candidate_masks = masks
    candidate_scores = scores

    # Initially select highest predicted score
    selected_candidate = int(np.argmax(scores))

    print("\nSAM predictions:")

    for i, score in enumerate(scores):

        print(
            f"Candidate {i + 1}: "
            f"Score={score:.4f}"
        )


# =====================================================
# 6. MOUSE CALLBACK
# =====================================================

def mouse_callback(event, x, y, flags, param):

    global points
    global labels

    # Left click = Positive foreground point
    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x, y])
        labels.append(1)

        print(f"Positive point: ({x}, {y})")

        run_prediction()

    # Right click = Negative background point
    elif event == cv2.EVENT_RBUTTONDOWN:

        points.append([x, y])
        labels.append(0)

        print(f"Negative point: ({x}, {y})")

        run_prediction()


# =====================================================
# 7. VISUALIZATION
# =====================================================

def create_display():

    display = image.copy()

    # Display selected SAM mask
    if candidate_masks is not None:

        mask = candidate_masks[selected_candidate]

        green = np.zeros_like(image)

        green[:] = (0, 255, 0)

        colored_mask = cv2.bitwise_and(
            green,
            green,
            mask=mask.astype(np.uint8) * 255
        )

        display = cv2.addWeighted(
            display,
            0.65,
            colored_mask,
            0.35,
            0
        )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            display,
            contours,
            -1,
            (0, 0, 255),
            2
        )

    # Draw positive and negative points
    for point, label in zip(points, labels):

        color = (
            (0, 255, 0)
            if label == 1
            else (0, 0, 255)
        )

        cv2.circle(
            display,
            tuple(point),
            6,
            color,
            -1
        )

    # Instructions
    instructions = [
        "LEFT CLICK: Positive (Egg)",
        "RIGHT CLICK: Negative (Background)",
        "1 / 2 / 3: Select candidate",
        "U: Undo last point",
        "R: Reset prompts",
        "S: Save mask",
        "Q: Quit"
    ]

    for i, instruction in enumerate(instructions):

        cv2.putText(
            display,
            instruction,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

    return display


# =====================================================
# 8. MAIN INTERACTIVE WINDOW
# =====================================================

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)

print("\n========== SAM PROMPT ANNOTATION ==========")

print("LEFT CLICK  : Egg foreground")
print("RIGHT CLICK : Background / tray / rod")
print("1 / 2 / 3   : Select candidate")
print("U           : Undo")
print("R           : Reset")
print("S           : Save")
print("Q           : Quit")


while True:

    display = create_display()

    cv2.imshow(
        window_name,
        display
    )

    key = cv2.waitKey(20) & 0xFF

    # Select candidate
    if key in [ord("1"), ord("2"), ord("3")]:

        candidate = int(chr(key)) - 1

        if candidate_masks is not None:

            selected_candidate = candidate

    # Undo
    elif key == ord("u"):

        if points:

            points.pop()
            labels.pop()

            run_prediction()

    # Reset
    elif key == ord("r"):

        points.clear()
        labels.clear()

        candidate_masks = None
        candidate_scores = None

        print("Prompts reset.")

    # Save
    elif key == ord("s"):

        if candidate_masks is None:

            print("Add at least one positive point first.")
            continue

        final_mask = (
            candidate_masks[selected_candidate]
            .astype(np.uint8) * 255
        )

        output_path = (
            OUTPUT_DIR /
            f"{Path(IMAGE_NAME).stem}.png"
        )

        cv2.imwrite(
            str(output_path),
            final_mask
        )

        overlay = create_display()

        cv2.imwrite(
            str(
                OUTPUT_DIR /
                f"{Path(IMAGE_NAME).stem}_overlay.jpg"
            ),
            overlay
        )

        print("\nSAM annotation saved!")
        print("Mask:", output_path)

        break

    # Quit
    elif key == ord("q"):

        print("Annotation cancelled.")
        break


cv2.destroyAllWindows()

print("\nProcess completed.")
