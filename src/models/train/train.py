import argparse
import os
import sys
import traceback

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import convert_labels

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run train YOLO model"
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
    "--resume",
    type=bool,
    default=False,
    help="Resume training choice: default False",
)
parser.add_argument(
    "--config",
    type=str,
    default="yolo8_baseline.yaml",
    help="Config model choice (default: yolo8_baseline.yaml)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="bdd100k.yaml",
    help="Dataset choice: default bdd100k.yaml",
)
parser.add_argument(
    "--epochs",
    type=str,
    default="100",
    help="Count of epochs: default 100",
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
DATA_DIR = os.path.join(
    os.getenv("HOME_DIR"), "config", "datasets", parse_args.dataset
)  # Default dataset_name.yaml or personal_dataset_name.yaml
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))
PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)


# Modify labels from .json to .txt
if not os.path.exists(PROCESSED_DIR):
    for split in ["train", "val", "test"]:
        convert_labels(
            os.path.join(PROCESSED_DIR, "labels", split),
            os.path.join(PROCESSED_DIR, "labels", split),
            args["selected_classes"],
            img_size=IMG_SIZE,
        )


# Select model based on choice
if parse_args.model == "1":
    model_path = args["model_name"]
elif parse_args.model == "2":
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
elif parse_args.model == "3":
    model_path = os.path.join(PROJECT_DIR, "optimized", "best_optimized.pt")
    PROJECT_DIR = os.path.join(PROJECT_DIR, "optimized")
elif parse_args.model not in ["1", "2", "3"]:
    model_path = parse_args.model
else:
    raise ValueError(
        "Invalid model choice."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")


# Load model
model = YOLO(model_path, task="detect", verbose=True)


# Train
try:
    results = model.train(
        data=DATA_DIR,
        project=PROJECT_DIR,
        epochs=int(parse_args.epochs),
        imgsz=IMG_SIZE[0],
        batch=8,
        resume=parse_args.resume,
        device=-1,
        patience=10,
        optimizer="AdamW",
        plots=True,
        hsv_h=0.015,  # HSV augmentation for variative lightness/contrast
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # Without rotation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,  # Perspective for distortions
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,  # Mosaic
        mixup=0.0,
        amp=False,
    )
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = traceback.extract_tb(exc_traceback)
    filename, line_number, func_name, text = tb[-1]
    print(f"Error occurred in file: {filename}")
    print(f"Line {line_number}: {text}")
    print(f"Error message: {e}")
