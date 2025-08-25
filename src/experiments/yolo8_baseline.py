import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import convert_labels

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
dataset_yaml = "bdd100k.yaml"
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    handle_model_yaml,
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Set directories path for training
PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)
DATA_DIR = os.path.join(
    os.getenv("HOME_DIR"), "config", "datasets", dataset_yaml
)  # Default dataset_name.yaml or personal_dataset_name.yaml
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))


# Modify labels from .json to .txt
for split in ["train", "val", "test"]:
    convert_labels(
        os.path.join(PROCESSED_DIR, "labels", split),
        os.path.join(PROCESSED_DIR, "labels", split),
        args["selected_classes"],
        img_size=IMG_SIZE,
    )


# Load model
resume = False
if not os.path.exists(PROJECT_DIR):
    model_path = args["model_name"]
else:
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
    resume = True
model = YOLO(model_path, task="detect", verbose=True)


# Train
try:
    results = model.train(
        data=DATA_DIR,
        project=PROJECT_DIR,
        epochs=100,
        imgsz=IMG_SIZE[0],
        batch=8,
        exist_ok=True,
        resume=resume,
        device=-1,
        patience=10,
        optimizer="AdamW",
        plots=True,
        amp=False,
    )
except Exception as e:
    print(f"Last training was finished: {e}.")
