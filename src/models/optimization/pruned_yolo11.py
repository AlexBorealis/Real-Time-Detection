import os

import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from torch.nn.utils import prune
from ultralytics import YOLO

load_dotenv()

# Set directory
os.chdir(os.getenv("HOME_DIR"))

# Set model.yaml path
# Create your yaml config file model
# model_name: /path/to/model_name.pt
# project_results_name: example_project
# optimized_project_results_name: example_project_optimized
# selected_classes: [class0, class1, class2, ..., classN]
handle_model_yaml = "yolo11_last_version.yaml"  # handle_model.yaml
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    handle_model_yaml,
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


# Load model
model_path = os.path.join(
    PROJECT_DIR,
    "train",
    "weights",
    "best.pt",
)
model = YOLO(model_path, task="detect", verbose=True)


# Pruning 0.1 weights
for name, module in model.model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.1)
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
