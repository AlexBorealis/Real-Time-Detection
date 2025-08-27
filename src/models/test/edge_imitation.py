import argparse
import os
import random

import onnxruntime as ort
import torch
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.metrics import fps

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run imitation YOLO model on the edge device"
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
    "--format",
    type=str,
    default="onnx",
    help="Model format: onnx, torchscript, engine (default: torchscript)",
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
dataset_yaml = "bdd100k.yaml"
yaml_path = os.path.join(
    os.getenv("HOME_DIR"),
    "config",
    "models",
    parse_args.config,
)
with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)


# Directories
DATA_DIR = os.path.join(
    os.getenv("HOME_DIR"), "config", "datasets", dataset_yaml
)  # Default dataset_name.yaml or personal_dataset_name.yaml
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))
PROJECT_DIR = os.path.join(
    os.getenv("HOME_DIR"), "results", "models", args["project_results_name"]
)
TESTING_IMG_DIR = os.path.join(
    os.getenv("HOME_DIR"), "data", "processed", "images", "test"
)  # Testing images directory


# Select model based on choice
if parse_args.model == "1":
    transform_model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.onnx")
    model_path = os.path.join(PROJECT_DIR, "train", "weights", "best.pt")
elif parse_args.model == "2":
    transform_model_path = os.path.join(PROJECT_DIR, "optimized", "best_optimized.onnx")
    model_path = os.path.join(PROJECT_DIR, "optimized", "best_optimized.pt")
elif parse_args.model not in ["1", "2"]:
    transform_model_path = parse_args.format
    model_path = parse_args.model
else:
    raise ValueError(
        "Invalid model choice. Use 1 for base model or 2 for optimized model."
    )

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")


# Load model
best_model = YOLO(model_path, task="detect", verbose=True).to(torch.device("cpu"))


# Measure model size
model_size = os.path.getsize(transform_model_path) / (1024 * 1024)
print(f"Model size: {model_size:.2f} MB")


# Load ONNX model on CPU
sess = ort.InferenceSession(transform_model_path, providers=["CPUExecutionProvider"])
image_path = os.path.join(
    TESTING_IMG_DIR,
    os.listdir(TESTING_IMG_DIR)[random.randint(0, len(os.listdir(TESTING_IMG_DIR)))],
)
result = fps(sess, image_path)
print(f"FPS on CPU (edge simulation): {round(result['fps'], 3)}")
print(f"Min time inference {round(min(result['times']) * 1000, 3)} ms")
print(
    f"Mean time inference {round(sum(result['times']) / len(result['times']) * 1000, 3)} ms"
)
print(f"Max time inference {round(max(result['times']) * 1000, 3)} ms")


# Get profile on CPU time
real_input_torch = torch.from_numpy(
    result["real_input"].transpose((0, 2, 3, 1))
).permute(0, 3, 1, 2)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    best_model(real_input_torch)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
