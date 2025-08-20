import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

MODEL_NAME = "yolov8n.pt"
RESULTS_DIR = os.path.join(os.getenv("HOME_DIR"), "results")
RESULTS_MODEL_NAME = "yolo_baseline"

model_path = os.path.join(
    os.getenv("HOME_DIR"), "src", "models", "initial_models", MODEL_NAME
)
model = YOLO(model_path, task="detect", verbose=True)

results = model.train(
    data=os.path.join(os.getenv("HOME_DIR"), "config", "bdd100k.yaml"),
    epochs=1,
    imgsz=320,
    project=os.path.join(RESULTS_DIR, "models"),
    name=RESULTS_MODEL_NAME,
    cache=False,
)
