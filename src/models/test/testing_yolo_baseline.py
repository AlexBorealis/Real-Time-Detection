import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

RESULTS_DIR = os.path.join(os.getenv("HOME_DIR"), "results")
TESTING_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed", "images", "test")

model_path = os.path.join(RESULTS_DIR, "models", "yolo_baseline", "weights", "best.pt")
model = YOLO(model_path, task="detect", verbose=True)

file_path = os.path.join(
    RESULTS_DIR,
    "visualizations",
    "yolo_baseline",
)
prediction = model.predict(
    source=os.path.join(TESTING_DIR, "cad7fdff-d9946f73.jpg"),
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    project=file_path,
    exist_ok=True,
)

