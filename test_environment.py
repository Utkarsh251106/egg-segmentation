
import torch
import cv2
import numpy as np
import matplotlib
import pandas as pd

from segment_anything import sam_model_registry

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)

print("CUDA available:", torch.cuda.is_available())
print("Device: CPU")

print("SAM imported successfully!")
print("Available SAM models:", list(sam_model_registry.keys()))