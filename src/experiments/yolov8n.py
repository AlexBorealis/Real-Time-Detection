import os

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

yaml_path = os.path.join(os.getenv("HOME_DIR"), "config", "models", "yolov8n.yaml")

with open(yaml_path, "r") as file:
    args = yaml.safe_load(file)

model_path = os.path.join(
    os.getenv("HOME_DIR"), "src", "models", "initial_models", args["model_name"]
)
model = YOLO(model_path, task="detect", verbose=True)

results = model.train(
    data=os.path.join(os.getenv("HOME_DIR"), "config", "datasets", "bdd100k.yaml"),
    epochs=1,
    imgsz=320,
    project=os.path.join(os.getenv("HOME_DIR"), "results", "models"),
    name=args["results_model_name"],
    cache=False,
    exist_ok=True,
)
