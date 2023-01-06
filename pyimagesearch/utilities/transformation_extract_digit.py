import cv2
import numpy as np
from skimage.segmentation import clear_border


def threshold_and_clear_border(cell: np.array) -> np.array:
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    return thresh


def get_mask(cnts: list[np.array], zero_shape: tuple) -> np.array:
    c = max(cnts, key=cv2.contourArea)
    mask: np.array = np.zeros(zero_shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def get_filled_percentage(thresh: np.array, mask: np.array) -> float:
    (h, w) = thresh.shape
    return cv2.countNonZero(mask) / float(w * h)
