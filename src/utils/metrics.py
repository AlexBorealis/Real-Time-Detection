import time

import numpy as np
from onnxruntime import InferenceSession

from src.utils.utils import load_image


# IOU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# mAP@0.5
def compute_map(pred_bboxes, true_bboxes, iou_threshold=0.5):
    ious = []
    for pred, true in zip(pred_bboxes, true_bboxes):
        iou = compute_iou(pred, true)
        ious.append(iou >= iou_threshold)

    precision = np.cumsum(ious) / (np.arange(len(ious)) + 1)
    recall = np.cumsum(ious) / len(true_bboxes)

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


# Compute FPS
def fps(sess: InferenceSession, image_path: str):
    input_name = sess.get_inputs()[0].name

    # Prepare real input
    real_input = np.expand_dims(
        load_image(image_path).transpose((2, 0, 1)) / 255.0, axis=0
    ).astype(np.float32)

    times = []
    for _ in range(100):
        start = time.time()
        _ = sess.run(None, {input_name: real_input})
        times.append(time.time() - start)
    avg_time = np.mean(times)

    return {
        "fps": 1 / avg_time,  # type: ignore
        "real_input": real_input
    }