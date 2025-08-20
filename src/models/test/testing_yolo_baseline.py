import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

yaml_path = os.path.join(os.getenv("HOME_DIR"), "config", "models", "yolov8n.yaml")

with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

TESTING_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed", "images", "test")

model_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "models",
    args["results_model_name"],
    "weights",
    "best.pt",
)
model = YOLO(model_path, task="detect", verbose=True)

file_path = os.path.join(
    os.getenv("HOME_DIR"),
    "results",
    "visualizations",
    args["results_model_name"],
)
prediction = model.predict(
    source=os.path.join(
        os.getenv("HOME_DIR"),
        "data",
        "processed",
        "images",
        "test",
        "cad7fdff-d9946f73.jpg",
    ),
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    project=file_path,
    exist_ok=True,
)
