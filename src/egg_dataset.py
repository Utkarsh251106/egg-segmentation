
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "train" / "egg"
MASK_DIR = ROOT / "outputs" / "batch_sam_masks" / "train" / "egg"

IMAGE_SIZE = 256
BATCH_SIZE = 4
SEED = 42


class EggSegmentationDataset(Dataset):
    def __init__(self, image_mask_pairs, image_size=256):
        self.pairs = image_mask_pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        # Read image and mask
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize image and mask
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert mask to binary: background=0, egg=1
        mask = (mask > 0).astype(np.float32)

        # Convert HWC to CHW for PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def get_image_mask_pairs():
    pairs = []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        mask_path = MASK_DIR / f"{image_path.stem}.png"

        if not mask_path.exists():
            print(f"Missing mask: {image_path.name}")
            continue

        pairs.append((image_path, mask_path))

    return pairs


def create_dataloaders():
    pairs = get_image_mask_pairs()

    random.seed(SEED)
    random.shuffle(pairs)

    # 80% training, 20% internal validation
    split_index = int(0.8 * len(pairs))

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    train_dataset = EggSegmentationDataset(train_pairs, IMAGE_SIZE)
    val_dataset = EggSegmentationDataset(val_pairs, IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    images, masks = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    print(f"Image values: {images.min():.2f} to {images.max():.2f}")
    print(f"Mask values: {torch.unique(masks)}")
