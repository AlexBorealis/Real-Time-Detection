import os
import random

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import generate_predicted_video

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
    os.getenv("HOME_DIR"), "config", "models", "yolo8_baseline.yaml"
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

# Set directories path for training
TESTING_VIDEO_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "raw", "videos", "BDDA", "test", "camera_videos"
)  # Test directory
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
    "videos",
)  # Result directory
VIDEO_NAME = os.listdir(TESTING_VIDEO_DIR)[
    random.randint(0, len(os.listdir(TESTING_VIDEO_DIR)))
]  # Video file name
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)  # Directory for saving results (logs, images, models)

# Load a model
if not os.path.exists(PROJECT_DIR):
    model_path = os.path.join(os.getenv("HOME_DIR"), "best_optimized.pt")
else:
    model_path = os.path.join(
        PROJECT_DIR,
        "train",
        "weights",
        "best.pt",
    )
model = YOLO(model_path, task="detect", verbose=True)

# Predict
generate_predicted_video(
    model,
    video_dir=TESTING_VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    video_name=VIDEO_NAME,
)
