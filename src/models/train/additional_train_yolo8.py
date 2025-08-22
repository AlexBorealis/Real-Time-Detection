import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Set directory
os.chdir(os.getenv("HOME_DIR"))

# Set model.yaml path
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    "yolo8_baseline.yaml",  # Change on different model
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

# Directories
PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["additional_project_results_name"]
)
DATA_DIR = os.path.join(os.getenv("HOME_DIR"), "config", "datasets", "bdd100k.yaml")

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

# Train
results = model.train(
    data=DATA_DIR,
    project=PROJECT_DIR,
    epochs=5,
    imgsz=os.getenv("HEIGHT"),
    batch=8,
    exist_ok=True,
    patience=10,
    optimizer="AdamW",
    plots=True,
    amp=False,
)
