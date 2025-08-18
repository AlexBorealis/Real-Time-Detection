import numpy as np


# IOU
def compute_iou(box1, box2):
    if not (len(box1) == 4 and len(box2) == 4):
        raise ValueError("Boxes must be [x1, y1, x2, y2]")

    if (
        box1[2] <= box1[0]
        or box1[3] <= box1[1]
        or box2[2] <= box2[0]
        or box2[3] <= box2[1]
    ):
        raise ValueError("Invalid box coordinates")

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
