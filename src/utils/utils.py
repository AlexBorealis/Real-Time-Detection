import json
import os
import random

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


def convert_labels(
    input_dir: str,
    output_dir: str,
    selected_classes: list[str],
    img_size: tuple = (320, 320),
):
    for label_file in os.listdir(input_dir):
        if label_file.endswith(".json"):
            img_name = label_file.replace(".json", "")
            with open(os.path.join(input_dir, label_file), "r") as f:
                data = json.load(f)
            frame = data.get("frames", [{}])[0]
            objects = frame.get("objects", [])
            with open(os.path.join(output_dir, f"{img_name}.txt"), "w") as f_out:
                for obj in objects:
                    if "box2d" in obj and obj["category"] in selected_classes:
                        class_id = selected_classes.index(obj["category"])
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


def generate_predicted_images(
    model,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    project_dir: str,
    num_images: int = 5,
    conf: float = 0.25,
    iou: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    # Select random images
    selected_images = random.sample(
        os.listdir(images_dir), min(num_images, len(os.listdir(images_dir)))
    )

    for img_name in selected_images:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(
            labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        # Predict
        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            save=True,
            save_txt=True,
            project=project_dir,
            exist_ok=True,
        )

        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Draw predicted boxes (red, thin, no labels)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Thin red box
                cv2.putText(
                    img,
                    "predict",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # Draw ground truth boxes (green, thin, no labels)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if (
                        len(parts) == 5
                    ):  # Expecting format: class x_center y_center width height
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        cv2.rectangle(
                            img, (x1, y1), (x2, y2), (0, 255, 0), 1
                        )  # Thin green box
                        cv2.putText(
                            img,
                            "real",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

        # Save
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)


def generate_predicted_video(
    model,
    video_dir: str,
    video_name: str,
    output_dir: str,
    conf: float = 0.25,
):
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(video_dir, video_name)
    video_path_out = os.path.join(output_dir, video_name.replace(".mp4", "_out.mp4"))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return

    h, w, _ = frame.shape
    out = cv2.VideoWriter(
        video_path_out,
        cv2.VideoWriter_fourcc(*"MP4V"),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (w, h),
    )

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > conf:
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4
                )
                cv2.putText(
                    frame,
                    results.names[int(class_id)].upper(),
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
