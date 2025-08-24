import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_images

load_dotenv()

# Set directory
os.chdir(os.getenv("HOME_DIR"))

# Set model.yaml path
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    "yolo11_last_version.yaml",  # Change on different model
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
VISUALIZE_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
)
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
    "comparison",
)

# Get model
model_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "models",
    args["project_results_name"],
    "train",
    "weights",
    "best.pt",
)
model = YOLO(model_path, task="detect", verbose=True)

generate_predicted_images(
    model,
    images_dir=TESTING_IMG_DIR,
    labels_dir=TESTING_LABEL_DIR,
    project_dir=VISUALIZE_DIR,
    output_dir=OUTPUT_DIR,
    num_images=20,
)
