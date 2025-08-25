import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_images

load_dotenv()

# Set directory
os.chdir(os.getenv("HOME_DIR"))

# Set model.yaml path
# Create your yaml config file model
# model_name: /path/to/model_name.pt
# project_results_name: example_project
# optimized_project_results_name: example_project_optimized
# selected_classes: [class0, class1, class2, ..., classN]
handle_model_yaml = "yolo8_baseline.yaml"  # handle_model.yaml
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    handle_model_yaml,
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Directories
TESTING_IMG_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "images", "test"
)  # Testing images directory
TESTING_LABEL_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "labels", "test"
)  # Testing labels directory
VISUALIZE_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
)  # Directory for visualizations
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
    "comparison",
)  # Result directory
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)  # Directory for saving results (logs, images, models)


# Load model
model_path = os.path.join(
    PROJECT_DIR,
    "optimized",
    "best_optimized.pt",
)
if not os.path.exists(model_path):
    model_path = os.path.join(
        PROJECT_DIR,
        "train",
        "weights",
        "best.pt",
    )
model = YOLO(model_path, task="detect", verbose=True)


# Predict
generate_predicted_images(
    model,
    images_dir=TESTING_IMG_DIR,
    labels_dir=TESTING_LABEL_DIR,
    project_dir=VISUALIZE_DIR,
    output_dir=OUTPUT_DIR,
    num_images=20,
)
