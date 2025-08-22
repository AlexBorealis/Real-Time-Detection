import os
import random

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Set directory
os.chdir(os.getenv("HOME_DIR"))

yaml_path = os.path.join(
    os.getenv("HOME_DIR"), "config", "models", "yolo8_baseline.yaml"
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

# Set directories path for testing
TESTING_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed", "images", "test")

# Get model
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

# Get image for testing
project_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["results_model_name"],
)
len_test_data = len(os.listdir(TESTING_DIR))
image_path = os.listdir(TESTING_DIR)[random.randrange(0, len_test_data)]

# Test
prediction = model.predict(
    source=os.path.join(
        TESTING_DIR,
        image_path,
    ),
    conf=0.25,
    iou=0.5,
    save=True,
    save_txt=True,
    project=project_path,
    exist_ok=True,
)
