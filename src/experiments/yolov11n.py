import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data/processed")
RESULTS_DIR = os.path.join(os.getenv("HOME_DIR"), "results", "models")
MODELS_DIR = os.path.join(os.getenv("HOME_DIR"), "src", "models")

model = YOLO(os.path.join(MODELS_DIR, "yolo11n.pt"), task='detect', verbose=True)

results = model.train(
    data=os.path.join(os.getenv("HOME_DIR"), 'config', "bdd100k.yaml"),
    epochs=1,
    imgsz=320,
    project=RESULTS_DIR,
    name="yolo_last_version",
    cache=False
)

metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")