import json
import os
import random

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from torch import cosine_similarity
from torchvision.ops import nms
from tqdm import tqdm


def postprocess_detections(
    detections: np.ndarray, iou: float = 0.5, ratio_median: float = 1.0
) -> np.ndarray:
    if len(detections) == 0:
        return detections

    areas = (detections[:, 2] - detections[:, 0]) * (
        detections[:, 3] - detections[:, 1]
    )
    median_area = np.median(areas)
    min_area = median_area * ratio_median  # type: ignore
    detections = detections[areas > min_area]

    if len(detections) == 0:
        return detections

    # NMS
    boxes = torch.tensor(detections[:, :4], dtype=torch.float32)
    scores = torch.tensor(detections[:, 4], dtype=torch.float32)
    keep_indices = nms(boxes, scores, iou)
    return detections[keep_indices.numpy()]


def to_centers(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, score = boxes.T
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.column_stack([xc, yc, w, h, score])


def similarity(tensor1: torch.Tensor, tensor2: torch.Tensor, params: tuple) -> float:
    iou, ratio = params
    detections1 = tensor1[:, :5].detach().clone().cpu().numpy()
    detections2 = tensor2[:, :5].detach().clone().cpu().numpy()
    processed = postprocess_detections(detections2, iou, ratio)

    if len(processed) == 0 or len(tensor1) == 0:
        return -1.0

    centers1 = to_centers(detections1)
    centers2 = to_centers(processed)

    dist = cdist(centers1[:, :2], centers2[:, :2])
    row_ind, col_ind = linear_sum_assignment(dist)

    matched1 = torch.from_numpy(centers1[row_ind]).float()
    matched2 = torch.from_numpy(centers2[col_ind]).float()

    if matched1.ndim == 1:
        matched1 = matched1.unsqueeze(0)
        matched2 = matched2.unsqueeze(0)

    sims = cosine_similarity(matched1, matched2)
    sims = sims.diagonal() if sims.ndim > 1 else sims
    return -float(torch.mean(sims))


def tune_parameters(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    initial: tuple[float, float] = (0.5, 1.0),
    iou_bounds: tuple[float, float] = (0.1, 0.9),
    ratio_bounds: tuple[float, float] = (0.1, 2.0),
) -> dict:
    """
    Подбирает оптимальные iou и ratio_median для максимального косинусного сходства
    между tensor1 и обработанным tensor2.

    Args:
        tensor1: torch.Tensor [N, 6] - детекции базовой модели
        tensor2: torch.Tensor [M, 6] - детекции оптимизированной модели
        iou_bounds: tuple - границы для iou
        ratio_bounds: tuple - границы для ratio_median

    Returns:
        dict: {'iou': opt_iou, 'ratio_median': opt_ratio, 'similarity': max_sim}
        :type initial: tuple[float, float]
    """

    def sim_wrapper(params: tuple) -> float:
        return similarity(tensor1, tensor2, params)

    res = minimize(
        sim_wrapper, initial, bounds=[iou_bounds, ratio_bounds], method="L-BFGS-B"
    )

    opt_iou, opt_ratio = res.x
    max_sim = -res.fun
    return {
        "iou": float(opt_iou),
        "ratio_median": float(opt_ratio),
        "similarity": float(max_sim),
    }


def load_image(img_path: str, rgb: bool = True) -> np.ndarray:
    if rgb:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return image


def load_data(img_path: str, label_path: str, rgb: bool = True):
    image = load_image(img_path) if rgb else load_image(img_path, rgb=False)

    labels = None
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = json.load(f)

    return image, labels


def convert_labels(
    input_dir: str,
    output_dir: str,
    selected_classes: list[str],
    img_size: tuple = (320, 320),
):
    for label_file in tqdm(os.listdir(input_dir), desc="Converting labels"):
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
            detections = result.boxes.data.cpu().numpy() # [x1, y1, x2, y2, score]
            filtered_detections = postprocess_detections(detections, iou=iou)

            # Draw predicted boxes (red, thin, no labels)
            for i, det in enumerate(filtered_detections):
                x1, y1, x2, y2, _, _ = map(int, det)
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
    iou: float = 0.1,
    ratio_median: float = 0.5,
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
        cv2.VideoWriter_fourcc(*"MP4V"),  # type: ignore
        int(cap.get(cv2.CAP_PROP_FPS)),
        (w, h),
    )

    while ret:
        results = model(frame)[0]

        boxes_data = (
            results.boxes.data.cpu().numpy()
        )  # [x1, y1, x2, y2, score, class_id]
        filtered_detections = postprocess_detections(
            boxes_data, iou=iou, ratio_median=ratio_median
        )

        # Draw predicted boxes
        for i, det in enumerate(filtered_detections):
            x1, y1, x2, y2, _, class_id = map(int, det)
            class_name = (
                results.names[class_id] if len(filtered_detections) > i else "predict"
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(
                frame,
                class_name.upper(),
                (x1, y1 - 10),
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


def sparsity(model):
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).to(torch.int).sum()  # type: ignore
    return b / a
