import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

RESULTS_DIR = os.path.join(os.getenv("HOME_DIR"), "results", "models")
MODELS_DIR = os.path.join(os.getenv("HOME_DIR"), "src", "models")
model_path = os.path.join(MODELS_DIR, "yolov8n.pt")

model = YOLO(model_path, task='detect', verbose=True)

results = model.train(
    data=os.path.join(os.getenv("HOME_DIR"), 'config', "bdd100k.yaml"),
    epochs=1,
    imgsz=320,
    project=RESULTS_DIR,
    name="yolo_baseline",
    cache=False
)