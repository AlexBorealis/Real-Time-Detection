import os
import random

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_images

load_dotenv()
os.chdir(os.getenv("HOME_DIR"))

yaml_path = os.path.join(
    os.getenv("HOME_DIR"), "config", "models", "yolo8_baseline.yaml"
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

# Directories
TESTING_IMG_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "images", "test"
)
TESTING_LABEL_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "labels", "test"
)

# Model
model_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "models",
    args["results_model_name"],
    "train",
    "weights",
    "best.pt",
)
model = YOLO(model_path, task="detect", verbose=True)

# Output dir
output_dir = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["results_model_name"],
    "comparison",
)
os.makedirs(output_dir, exist_ok=True)

# Number of images
num_images = 5

# Select random images
image_files = os.listdir(TESTING_IMG_DIR)
selected_images = random.sample(image_files, min(num_images, len(image_files)))

generate_predicted_images(
    model,
    images_dir=TESTING_IMG_DIR,
    labels_dir=TESTING_LABEL_DIR,
    output_dir=output_dir,
    num_images=num_images,
)
