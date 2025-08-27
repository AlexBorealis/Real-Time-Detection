import argparse
import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_images

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run image prediction with selected YOLO model"
)
parser.add_argument(
    "--nimage", type=int, default=1, help="Number of images to process (default: 1)"
)
parser.add_argument(
    "--model",
    type=str,
    default="1",
    help="""
    Model choice: 
    1 for init model (default value), 
    2 for trained model, 
    3 for optimized model, 
    'path/to/model.pt' for custom path
    """,
)
parser.add_argument(
    "--config",
    type=str,
    default="yolo8_baseline.yaml",
    help="Config model choice (default: yolo8_baseline.yaml)",
)
parse_args = parser.parse_args()
load_dotenv()


# Set directory
os.chdir(os.getenv("HOME_DIR"))


# Set model.yaml path
# Create your yaml config file model
# model_name: /path/to/model_name.pt
# project_results_name: example_project
# optimized_project_results_name: example_project_optimized
# selected_classes: [class0, class1, class2, ..., classN]
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    parse_args.config,
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Directories
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
)  # Result directory
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)  # Directory for saving results (logs, images, models)
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
)  # Directory for visualizations


# Select model based on choice
if parse_args.model == "1":
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
    VISUALIZE_DIR = os.path.join(VISUALIZE_DIR, args["project_results_name"])
    OUTPUT_DIR = os.path.join(VISUALIZE_DIR, args["project_results_name"], "comparison")
elif parse_args.model == "2":
    model_path = os.path.join(PROJECT_DIR, "optimized", "train", "weights", "best.pt")
    VISUALIZE_DIR = os.path.join(VISUALIZE_DIR, "yolo8_baseline_optimized")
    OUTPUT_DIR = os.path.join(VISUALIZE_DIR, "yolo8_baseline_optimized", "comparison")
elif parse_args.model not in ["1", "2"]:
    model_path = parse_args.model
    VISUALIZE_DIR = os.path.join(VISUALIZE_DIR, "custom_model")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "custom_model", "comparison")
else:
    raise ValueError(
        "Invalid model choice. Use 1 for base model or 2 for optimized model."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")


# Load model
model = YOLO(model_path, task="detect", verbose=True)


# Predict
generate_predicted_images(
    model,
    images_dir=TESTING_IMG_DIR,
    labels_dir=TESTING_LABEL_DIR,
    project_dir=VISUALIZE_DIR,
    output_dir=OUTPUT_DIR,
    num_images=parse_args.nimage,
)
