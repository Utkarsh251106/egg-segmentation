"""
Interactive SAM batch annotation (recommended: TRAIN split first).

Controls:
  Left click  = positive point inside egg
  Right click = negative point on rod/background
  1 / 2 / 3    = choose one of SAM's candidate masks
  U            = undo last point
  R            = reset points
  S            = save mask and advance
  N            = skip image
  Q / ESC      = quit

Run from project root:
  python src/batch_sam_annotation.py

Expected:
  dataset/test/egg/*.jpg
  models/sam_vit_b_01ec64.pth

Masks are saved as:
  outputs/test_predictions/ground_truth/<image-stem>.png
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "test" / "egg"
CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
OUTPUT_DIR = ROOT / "outputs" / "test_predictions" / "ground_truth"
MODEL_TYPE = "vit_b"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW = "SAM Egg Annotation | Left=egg, Right=not egg"

points = []
labels = []
candidate_masks = None
candidate_scores = None
selected_candidate = 0
image_rgb = None
image_path = None
predictor = None


def predict():
    global candidate_masks, candidate_scores, selected_candidate

    if not points:
        candidate_masks = None
        candidate_scores = None
        selected_candidate = 0
        return

    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    candidate_masks = masks
    candidate_scores = scores
    selected_candidate = int(np.argmax(scores))


def selected_mask():
    if candidate_masks is None:
        return None
    return candidate_masks[selected_candidate].astype(bool)


def redraw():
    if image_rgb is None:
        return

    frame = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    mask = selected_mask()

    if mask is not None:
        overlay = np.zeros_like(frame)
        overlay[mask] = (40, 220, 40)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        cv2.circle(frame, (int(x), int(y)), 9, (255, 255, 255), 1)

    score = ""
    if candidate_scores is not None:
        score = f" | Candidate {selected_candidate + 1}/3 | score {candidate_scores[selected_candidate]:.3f}"

    lines = [
        f"{image_path.name}{score}",
        "Left click: egg (+) | Right click: non-egg (-)",
        "1/2/3: candidate | U: undo | R: reset",
        "S: save+next | N: skip | Q/ESC: quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 24

    cv2.imshow(WINDOW, frame)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        labels.append(1)
        predict()
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        labels.append(0)
        predict()
        redraw()


def reset():
    points.clear()
    labels.clear()
    predict()
    redraw()


def load_image(path):
    global image_rgb, image_path
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"Could not read: {path}")
        return False
    image_path = path
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    reset()
    return True


def main():
    global predictor, selected_candidate

    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"SAM checkpoint not found: {CHECKPOINT}")

    files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not files:
        raise SystemExit(f"No images found in: {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM {MODEL_TYPE} on {device}...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    saved = 0
    already_present = 0

    for i, path in enumerate(files):
        out_path = OUTPUT_DIR / f"{path.stem}.png"
        if out_path.exists():
            print(f"[{i + 1}/{len(files)}] Existing mask, skipping: {path.name}")
            already_present += 1
            continue

        if not load_image(path):
            continue

        print(f"[{i + 1}/{len(files)}] {path.name}")
        redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("1"), ord("2"), ord("3")):
                choice = key - ord("1")
                if candidate_masks is not None and choice < len(candidate_masks):
                    selected_candidate = choice
                    redraw()

            elif key in (ord("u"), ord("U")):
                if points:
                    points.pop()
                    labels.pop()
                    predict()
                    redraw()

            elif key in (ord("r"), ord("R")):
                reset()

            elif key in (ord("s"), ord("S")):
                mask = selected_mask()
                if mask is None:
                    print("Add at least one prompt point before saving.")
                    continue
                binary = mask.astype(np.uint8) * 255
                if binary.shape != image_rgb.shape[:2]:
                    binary = cv2.resize(
                        binary, (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                if cv2.imwrite(str(out_path), binary):
                    print(f"Saved: {out_path}")
                    saved += 1
                else:
                    print(f"Failed to save: {out_path}")
                break

            elif key in (ord("n"), ord("N")):
                print(f"Skipped: {path.name}")
                break

            elif key in (ord("q"), ord("Q"), 27):
                cv2.destroyAllWindows()
                print(f"Stopped. New masks saved: {saved}; existing masks skipped: {already_present}")
                return

    cv2.destroyAllWindows()
    print("\nAnnotation complete.")
    print(f"New masks saved: {saved}")
    print(f"Existing masks skipped: {already_present}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
