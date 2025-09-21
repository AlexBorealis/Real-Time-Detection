import argparse
import os
import random

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_video

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run video prediction with selected YOLO model"
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


# Set directories path for training
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
)  # Result directory
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)  # Directory for saving results (logs, images, models)
TESTING_VIDEO_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "raw", "videos", "BDDA", "test", "camera_videos"
)  # Test directory
VIDEO_NAME = os.listdir(TESTING_VIDEO_DIR)[
    random.randint(0, len(os.listdir(TESTING_VIDEO_DIR)))
]  # Video file name


# Select model based on choice
if parse_args.model == "1":
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, args["project_results_name"], "videos")
elif parse_args.model == "2":
    model_path = os.path.join(PROJECT_DIR, "optimized", "train3", "weights", "best.pt")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "yolo8_baseline_optimized", "videos")
elif parse_args.model not in ["1", "2"]:
    model_path = parse_args.model
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "custom_model", "videos")
else:
    raise ValueError(
        "Invalid model choice."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")


# Load model
model = YOLO(model_path, task="detect", verbose=True)


# Predict
generate_predicted_video(
    model,
    video_dir=TESTING_VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    video_name=VIDEO_NAME,
)
