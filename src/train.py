
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from egg_dataset import create_dataloaders
from model import build_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "best_egg_unet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
LEARNING_RATE = 1e-4


def dice_score(predictions, targets, smooth=1e-6):
    predictions = (torch.sigmoid(predictions) > 0.5).float()

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()

    return (2 * intersection + smooth) / (union + smooth)


def iou_score(predictions, targets, smooth=1e-6):
    predictions = (torch.sigmoid(predictions) > 0.5).float()

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(loader, desc="Validation"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()

    count = len(loader)

    return (
        total_loss / count,
        total_dice / count,
        total_iou / count
    )


def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = create_dataloaders()

    model = build_model().to(DEVICE)

    # BCE + Dice Loss
    bce = nn.BCEWithLogitsLoss()

    def criterion(outputs, targets):
        bce_loss = bce(outputs, targets)

        probabilities = torch.sigmoid(outputs)

        intersection = (probabilities * targets).sum()
        denominator = probabilities.sum() + targets.sum()

        dice_loss = 1 - (
            (2 * intersection + 1e-6)
            / (denominator + 1e-6)
        )

        return bce_loss + dice_loss

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dice = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print("Best model saved.")

    print(f"\nBest Validation Dice: {best_dice:.4f}")
    print(f"Model saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
