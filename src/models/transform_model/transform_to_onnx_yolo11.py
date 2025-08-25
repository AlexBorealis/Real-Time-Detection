import os

import yaml
from dotenv import load_dotenv
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
    handle_model_yaml,  # handle_model.yaml
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Directories
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)
TESTING_IMG_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "images", "test"
)  # Testing images directory

# Load model
model_path = os.path.join(
    PROJECT_DIR,
    "optimized",
    "best_optimized.pt",
)
if not os.path.exists(model_path):
    model_path = os.path.join(
        PROJECT_DIR,
        "train",
        "weights",
        "best.pt",
    )

model = YOLO(model_path, task="detect", verbose=True)
onnx_path = model.export(
    format="onnx",
    imgsz=IMG_SIZE[0],
    dynamic=False,
)
