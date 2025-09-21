import argparse
import os

import torch
import yaml
from dotenv import load_dotenv
from torch.quantization import quantize_dynamic
from ultralytics import YOLO

# Parse arguments
parser = argparse.ArgumentParser(
    description="Quantization model"
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


# Quantization to int8
model = quantize_dynamic(
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

# Saving model with metadata
yolo_model = YOLO(model_path, task="detect", verbose=True)  # Use original YAML config
yolo_model.load_state_dict(model.state_dict())  # Load quantized weights

# Save quantized model with metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)
quantized_path = os.path.join(OUTPUT_DIR, "best_optimized.pt")
yolo_model.save(quantized_path)  # Save with YOLO metadata