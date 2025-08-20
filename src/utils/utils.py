import json
import os

import cv2


def load_data(img_path: str, label_path: str, rgb: bool = True):
    if rgb:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    with open(label_path, "r") as f:
        labels = json.load(f)

    return image, labels


def convert_labels(input_dir: str, output_dir: str, img_size: tuple = (320, 320)):
    for label_file in os.listdir(input_dir):
        if label_file.endswith(".json"):
            img_name = label_file.replace(".json", "")
            with open(os.path.join(input_dir, label_file), "r") as f:
                data = json.load(f)
            frame = data.get("frames", [{}])[0]
            objects = frame.get("objects", [])
            with open(os.path.join(output_dir, f"{img_name}.txt"), "w") as f_out:
                for obj in objects:
                    if "box2d" in obj and obj["category"] in [
                        "person",
                        "car",
                        "truck",
                        "bus",
                        "train",
                        "motorcycle",
                        "bicycle",
                        "rider",
                    ]:
                        class_id = [
                            "person",
                            "car",
                            "truck",
                            "bus",
                            "train",
                            "motorcycle",
                            "bicycle",
                            "rider",
                        ].index(obj["category"])
                        x1, y1, x2, y2 = (
                            obj["box2d"]["x1"],
                            obj["box2d"]["y1"],
                            obj["box2d"]["x2"],
                            obj["box2d"]["y2"],
                        )
                        x_center = (x1 + x2) / (2 * img_size[0])
                        y_center = (y1 + y2) / (2 * img_size[1])
                        width = (x2 - x1) / img_size[0]
                        height = (y2 - y1) / img_size[1]
                        f_out.write(
                            f"{class_id} {x_center} {y_center} {width} {height}\n"
                        )