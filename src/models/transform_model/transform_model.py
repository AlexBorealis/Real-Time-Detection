import argparse
import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

# Parse arguments
parser = argparse.ArgumentParser(
    description="Transform YOLO model to special format"
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
    "--format",
    type=str,
    default="torchscript",
    help="Model format: onnx, torchscript, engine (default: torchscript)",
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
    parse_args.config,  # handle_model.yaml
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Directories
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)


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

# Export
onnx_path = model.export(
    format="onnx",
    imgsz=IMG_SIZE[0],
    dynamic=False,
)
