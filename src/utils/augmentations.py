import os
import cv2
import json

from .metrics import compute_iou


# noinspection PyTypeChecker
def process_image(
    img_path,
    label_path,
    label_file,
    preprocessed_images_dir,
    preprocessed_labels_dir,
    transform,
    new_h=320,
    new_w=320,
):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "r") as f:
        labels = json.load(f)

    # Augmentation
    augmented = transform(image=img)
    augmented_img = augmented["image"]
    replay = augmented["replay"]

    # Rescaling bb
    h, w = img.shape[:2]
    scale_h, scale_w = new_h / h, new_w / w

    # Checking augmentations
    is_flipped = False
    holes = []
    for t in replay["transforms"]:
        if t["__class_fullname__"].endswith("HorizontalFlip") and t["applied"]:
            is_flipped = True
        if t["__class_fullname__"].endswith("CoarseDropout") and t["applied"]:
            holes = t["params"].get("holes", [])

    if "frames" in labels and labels["frames"]:
        for frame in labels["frames"]:
            new_objects = []
            for obj in frame.get("objects", []):
                if "box2d" in obj:
                    bbox = obj["box2d"]
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    x1_new = int(x1 * scale_w)
                    y1_new = int(y1 * scale_h)
                    x2_new = int(x2 * scale_w)
                    y2_new = int(y2 * scale_h)

                    if is_flipped:
                        x1_temp = x1_new
                        x1_new = new_w - x2_new
                        x2_new = new_w - x1_temp

                    bb = [x1_new, y1_new, x2_new, y2_new]

                    exclude = False
                    for hole in holes:
                        if compute_iou(bb, hole) > 0.5:
                            exclude = True
                            break
                    if not exclude:
                        obj["box2d"] = {
                            "x1": x1_new,
                            "y1": y1_new,
                            "x2": x2_new,
                            "y2": y2_new,
                        }
                        new_objects.append(obj)

            frame["objects"] = new_objects

    # Saving results
    img_file = os.path.basename(img_path)
    cv2.imwrite(
        os.path.join(preprocessed_images_dir, img_file),
        cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR),
    )
    with open(os.path.join(preprocessed_labels_dir, label_file), "w") as f:
        json.dump(labels, f, indent=4)
