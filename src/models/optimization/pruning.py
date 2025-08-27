import argparse
import os
import sys
import traceback

import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from torch.nn.utils import prune
from ultralytics import YOLO

# Parse arguments
parser = argparse.ArgumentParser(
    description="Pruning model"
)
parser.add_argument(
    "--model",
    type=str,
    default="1",
    help="""
    Model choice: 
    1 for trained model, 
    2 for optimized model, 
    'path/to/model.pt' for custom path
    """,
)
parser.add_argument(
    "--config",
    type=str,
    default="yolo8_baseline.yaml",
    help="Config model choice (default: yolo8_baseline.yaml)",
)
parser.add_argument(
    "--ratio",
    type=float,
    default=0.1,
    help="Ratio pruning weights (default: 0.1)",
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
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))
PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)  # Directory for saving results (logs, images, models)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "optimized",
)  # Result directory


# Select model based on choice
if parse_args.model == "1":
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
elif parse_args.model == "2":
    model_path = os.path.join(PROJECT_DIR, "optimized", "best_optimized.pt")
elif parse_args.model not in ["1", "2"]:
    model_path = parse_args.model
else:
    raise ValueError(
        "Invalid model choice."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")


# Load model
model = YOLO(model_path, task="detect", verbose=True)


# Pruning 0.1 weights
try:
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=parse_args.ratio)
            prune.remove(module, "weight")


    # Saving model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt = {
        "model": model.model,
        "train_args": {},
    }
    torch.save(
        ckpt,
        os.path.join(
            OUTPUT_DIR,
            "best_optimized.pt",
        ),
    )
    print("Pruning completed")
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = traceback.extract_tb(exc_traceback)
    filename, line_number, func_name, text = tb[-1]
    print(f"Error occurred in file: {filename}")
    print(f"Line {line_number}: {text}")
    print(f"Error message: {e}")