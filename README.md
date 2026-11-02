# Egg Segmentation using U-Net and ResNet34

A deep learning-based image segmentation project designed to automatically detect and segment eggs from images using a U-Net architecture with a ResNet34 encoder.

The project uses the Segment Anything Model (SAM) for annotation assistance and a trained U-Net model for egg segmentation. It also includes a Streamlit web application for interactive inference.

## Features

- Automatic egg segmentation from images
- U-Net architecture with ResNet34 encoder
- SAM-assisted image annotation
- Binary mask generation
- Prediction overlay visualization
- Egg coverage percentage calculation
- Downloadable segmentation masks
- Interactive Streamlit application
- Evaluation using Dice Score, IoU, Precision, Recall, and Pixel Accuracy

## Model Architecture

The segmentation model is based on:

- **Architecture:** U-Net
- **Encoder:** ResNet34
- **Encoder Weights:** ImageNet pretrained
- **Output Classes:** 1
- **Task:** Binary Image Segmentation
- **Input Image Size:** 256 × 256

U-Net is used for pixel-level segmentation, while ResNet34 extracts image features through its encoder.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Segmentation Models PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Streamlit
- Segment Anything Model (SAM)

## Project Structure

```text
egg-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── egg_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── visualize_evaluation.py
│   └── inference.py
│
├── models/
│   ├── best_egg_unet.pth
│   └── sam_vit_b_01ec64.pth
│
├── dataset/
│   ├── train/
│   │   └── egg/
│   ├── validation/
│   │   └── egg/
│   └── test/
│       └── egg/
│
└── outputs/
    ├── batch_sam_masks/
    └── test_predictions/
        ├── ground_truth/
        ├── predictions/
        ├── overlays/
        └── evaluation_results.csv
```

> Note: Dataset and model checkpoint files are not included in the public repository if they are excluded through `.gitignore`.

## Dataset

The dataset contains images of eggs used for training, validation, and testing.

### Dataset Preparation

The training dataset contains 75 annotated images.

These images were annotated using the Segment Anything Model (SAM), generating binary segmentation masks.

The dataset loader divides the 75 annotated training images into:

| Dataset | Images |
|---|---:|
| Training | 60 |
| Validation | 15 |
| Total | 75 |

The test dataset contains 11 images with manually reviewed SAM-generated ground-truth masks.

### Dataset Format

Images and masks are organized in corresponding directories.

```text
dataset/
├── train/
│   └── egg/
├── validation/
│   └── egg/
└── test/
    └── egg/
```

The corresponding segmentation masks represent the egg regions in binary format.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/egg-segmentation.git
```

Navigate to the project directory:

```bash
cd egg-segmentation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Checkpoints

Place the trained U-Net checkpoint in:

```text
models/best_egg_unet.pth
```

If you plan to run SAM annotation, place the SAM checkpoint in:

```text
models/sam_vit_b_01ec64.pth
```

The trained U-Net checkpoint is required for inference and the Streamlit application.

## Training

The model is trained using the annotated egg dataset.

Run:

```bash
python src/train.py
```

The best-performing model checkpoint is saved as:

```text
models/best_egg_unet.pth
```

The model learns to distinguish egg pixels from the background using binary segmentation.

## Prediction

To generate segmentation predictions:

```bash
python src/predict.py
```

The generated outputs include:

- Binary segmentation masks
- Prediction overlays
- Visual comparison of original images and predicted masks

The output images are saved in:

```text
outputs/test_predictions/
```

## Evaluation

The model is evaluated using the manually reviewed ground-truth masks of 11 test images.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Dice Score | 0.9675 |
| IoU | 0.9388 |
| Precision | 0.9465 |
| Recall | 0.9917 |
| Pixel Accuracy | 0.9945 |

### Metric Interpretation

- **Dice Score:** Measures overlap between predicted and ground-truth egg regions.
- **IoU:** Measures the intersection over union of predicted and actual masks.
- **Precision:** Measures the proportion of predicted egg pixels that are correct.
- **Recall:** Measures the proportion of actual egg pixels detected by the model.
- **Pixel Accuracy:** Measures the percentage of correctly classified pixels.

The model demonstrates strong segmentation performance on the evaluated test images.

> These results are based on 11 test images and should not be interpreted as a guarantee of performance on all egg images or imaging conditions.

Run evaluation:

```bash
python src/evaluate.py
```

The detailed evaluation results are saved to:

```text
outputs/test_predictions/evaluation_results.csv
```

## Streamlit Application

The project includes an interactive web application for performing egg segmentation.

### Features

- Upload an image
- Generate an egg segmentation mask
- Display the original image
- Display the predicted mask
- Display the segmentation overlay
- Calculate egg pixel count
- Calculate egg coverage percentage
- Download the generated mask
- Download the overlay image

### Run the Application

```bash
streamlit run app.py
```

The application opens in your browser.

## Inference Example

The inference module can be used independently of Streamlit.

```python
from src.inference import EggSegmenter

segmenter = EggSegmenter(
    model_path="models/best_egg_unet.pth"
)

mask = segmenter.predict("sample_egg.jpg")

overlay = segmenter.create_overlay(
    "sample_egg.jpg",
    mask
)
```

The predicted mask identifies the egg region in the input image.

## Future Improvements

- Increase the size and diversity of the dataset
- Improve segmentation around overlapping eggs
- Add egg counting functionality
- Support batch image processing
- Add video-based egg segmentation
- Deploy the application as a web service
- Explore YOLO-based segmentation architectures
- Improve model generalization under different lighting conditions

## Applications

Potential applications include:

- Food quality inspection
- Agricultural image analysis
- Automated egg counting
- Laboratory image analysis
- Computer vision research
- Automated visual inspection systems

## Author

**Sonu Kumar**

Computer Vision Engineer

## License

This project is intended for educational and research purposes.

A license can be added to define the terms under which this project may be used, modified, and distributed.
