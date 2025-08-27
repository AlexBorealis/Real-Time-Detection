import argparse
import gc
import os
import time

from albumentations import (
    CoarseDropout,
    GaussNoise,
    ReplayCompose,
    Resize,
)
from dotenv import load_dotenv, set_key
from tqdm import tqdm

from src.utils.augmentations import process_image

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run preprocessing data"
)
parser.add_argument(
    "--height",
    type=str,
    default="480",
    help="Height image: default 480",
)
parser.add_argument(
    "--width",
    type=str,
    default="480",
    help="Width image: default 480",
)
parse_args = parser.parse_args()
load_dotenv()

start = time.time()

# Paths
PROCESSED_DIR = os.path.join(os.getenv("HOME_DIR"), "data", "processed")
ENV_PATH = os.path.join(os.getenv("HOME_DIR"), ".env")
set_key(ENV_PATH, "HEIGHT", parse_args.height)
set_key(ENV_PATH, "WIDTH", parse_args.width)
IMG_SIZE = int(os.getenv("HEIGHT")), int(os.getenv("WIDTH"))

# Augmentation Pipelines
transform = ReplayCompose(
    [
        Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
        GaussNoise(p=0.5, std_range=(0.1, 0.1)),
        CoarseDropout(p=0.5, num_holes_range=(1, 3), hole_height_range=(8, 32)),
    ]
)

# Processing
for split in ["train", "test", "val"]:
    images_dir = os.path.join(os.getenv("RAW_IMAGES_DIR"), split)
    labels_dir = os.path.join(os.getenv("RAW_LABELS_DIR"), split)

    os.makedirs(os.path.join(PROCESSED_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "labels", split), exist_ok=True)
    processed_images_dir = os.path.join(PROCESSED_DIR, "images", split)
    processed_labels_dir = os.path.join(PROCESSED_DIR, "labels", split)

    selected_img_files = os.listdir(images_dir)

    batch_size_processing = 10000
    for i in range(0, len(selected_img_files), batch_size_processing):
        # noinspection PyRedeclaration
        batch_files = selected_img_files[i : i + batch_size_processing]
        for img_file in tqdm(
            batch_files,
            desc=f"Processing {split} batch {i // batch_size_processing + 1}",
        ):
            img_path = os.path.join(images_dir, img_file)
            label_file = img_file.replace(".jpg", ".json").replace(".png", ".json")
            label_path = os.path.join(labels_dir, label_file)

            process_image(
                img_path,
                label_path,
                processed_images_dir,
                processed_labels_dir,
                transform,
                new_h=IMG_SIZE[0],
                new_w=IMG_SIZE[1],
            )

        # Clearing memory after batch
        batch_files = None
        gc.collect()

print(f"Time {time.time() - start}")
