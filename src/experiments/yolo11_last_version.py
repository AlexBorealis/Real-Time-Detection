import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.utils import convert_labels

load_dotenv()

PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
yaml_path = os.path.join(
    os.getenv("HOME_DIR"), "config", "models", "yolo11_last_version.yaml"
)

with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)
for split in ["train", "val", "test"]:
    convert_labels(
        os.path.join(PROCESSED_DIR, "labels", split),
        os.path.join(PROCESSED_DIR, "labels", split),
        args["selected_classes"],
    )

model = YOLO(args["model_name"], task="detect", verbose=True)

results = model.train(
    data=os.path.join(os.getenv("HOME_DIR"), "config", "datasets", "bdd100k.yaml"),
    project=os.path.join(os.getenv("HOME_DIR"), "results", "models"),
    epochs=100,
    imgsz=320,
    cache=True,
    exist_ok=True,
    device=-1,
    resume=True,
    patience=5,
    batch=-1,
    optimizer="AdamW",
    classes=[0, 1, 5, 6, 7],
    cos_lr=True,
    plots=True,
)