import cv2
import imutils
import numpy as np


def load_image(path: str) -> np.array:
    image = cv2.imread(path)
    return imutils.resize(image, width=600)
