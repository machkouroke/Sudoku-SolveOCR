import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform


def threshold_and_blur(image: np.array) -> tuple[np.array, np.array, np.array]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # appliquer un seuillage adaptatif, puis inverser la carte de seuil
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    return gray, blurred, thresh


def find_contours(thresh: np.array) -> list[np.array]:
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts: list = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts


def find_puzzle_cont(cnts: list[np.array]) -> np.array:
    puzzleCnt = None
    # boucle sur les contours
    for c in cnts:
        # approximer le contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # si notre contour approximatif a quatre points,
        # alors nous pouvons supposer que nous avons trouvé le contour du puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break
    return puzzleCnt


def resize_to_grid(image: np.array, gray: np.array, puzzleCnt: np.array) -> tuple[np.array, np.array]:
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    return puzzle, warped
