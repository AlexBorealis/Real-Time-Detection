import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

MODEL_NAME = "yolo_baseline"
RESULTS_DIR = os.path.join(os.getenv("HOME_DIR"), "results", "models")
TESTING_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed", "images", "test")
BEST_MODELS_DIR = os.path.join(os.getenv("HOME_DIR"), "results", "models")
model_path = os.path.join(BEST_MODELS_DIR, MODEL_NAME, "weights", "best.pt")

model = YOLO(model_path, task='detect', verbose=True)

metrics = model.val(
    data=os.path.join(os.getenv("HOME_DIR"), 'config', "bdd100k.yaml"),
    project=RESULTS_DIR,
    name="val_baseline"
)