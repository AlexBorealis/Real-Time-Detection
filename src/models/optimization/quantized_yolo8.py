import os

import torch
import yaml
from dotenv import load_dotenv
from torch.quantization import quantize_dynamic
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
handle_model_yaml = "yolo8_baseline.yaml"  # handle_model.yaml
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    handle_model_yaml,
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Set directories path for training
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


# Quantization to int8
quantized_model = quantize_dynamic(
    model.model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
)


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
