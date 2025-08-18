from numpy.core.multiarray import array
from skimage.feature import hog
import cv2
import json


def load_data(img_path: str, label_path: str, rgb: bool = True):
    if rgb:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    with open(label_path, "r") as f:
        labels = json.load(f)

    return image, labels


def extract_hog(img: array):
    return hog(
        img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
    )
