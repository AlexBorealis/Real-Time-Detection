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
yaml_path = os.path.join(
    os.getenv("HOME_DIR"), "config", "models", "yolo11_last_version.yaml"
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

# Set directories path for training
VIDEO_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "raw", "videos", "BDDA", "test", "camera_videos"
)
OUTPUT_DIR = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["project_results_name"],
    "videos",
)
VIDEO_NAME = os.listdir(VIDEO_DIR)[random.randint(0, len(os.listdir(VIDEO_DIR)))]

# Load a model
model_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "models",
    args["project_results_name"],
    "train",
    "weights",
    "best.pt",
)
model = YOLO(model_path)  # load a custom model

# Predict
generate_predicted_video(
    model,
    video_dir=VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    video_name=VIDEO_NAME,
)