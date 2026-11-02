
import torch
import segmentation_models_pytorch as smp


def build_model():
    """
    U-Net with a ResNet34 encoder pretrained on ImageNet.
    Output is one channel for binary egg segmentation.
    """

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )

    return model


if __name__ == "__main__":
    model = build_model()

    # Test model with a dummy input
    dummy_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model created successfully.")
