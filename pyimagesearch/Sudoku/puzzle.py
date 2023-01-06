import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

from pyimagesearch.utilities.transformation_extract_digit import threshold_and_clear_border, get_mask, \
    get_filled_percentage
from pyimagesearch.utilities.transformation_find_puzzle import threshold_and_blur, find_contours, find_puzzle_cont, \
    resize_to_grid


def find_puzzle(image: np.array, debug: bool = False) -> tuple:
    # sourcery skip: raise-specific-error
    """
    Trouver la grille de Sudoku dans une image.
    :param image: Image de Sudoku.
    :param debug: Paramètre booléen pour activer la visualisation de chaque étape de la chaîne de traitement
    (Utile pour le débogage)
    :return: tuple (image originale avec la grille de Sudoku entourée, image en noir et blanc de la grille de Sudoku)
    """
    # Conversion de l'image en échelle de gris
    gray, blurred, thresh = threshold_and_blur(image)

    # Debuggage du seuillage
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # trouver des contours dans l'image seuillée et les trier par taille dans l'ordre décroissant
    cnts: list[np.array] = find_contours(thresh)

    # trouver le contour de la grille de Sudoku
    puzzleCnt: np.array = find_puzzle_cont(cnts)

    # Erreur en cas d'absence de contour
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # Vérification de l'exactitude du contour
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # appliquez une transformation de perspective pour recadrer l'image de la grille de Sudoku et l'image
    # en noir et blanc
    puzzle, puzzle_gray = resize_to_grid(image, gray, puzzleCnt)

    # vérifier si nous visualisons l'image recadrée
    if debug:
        # afficher l'image de sortie déformée (encore une fois, à des fins de débogage)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    return puzzle, puzzle_gray


def extract_digit(cell: np.array, debug: bool = False) -> np.array:
    """
    For a given cell, extract the digit (if one exists) from the cell.
    :param cell:
    :param debug:
    :return:
    """
    # Seuillage et suppression des bordures de la case
    thresh: np.array = threshold_and_clear_border(cell)

    # Debuggage de l'étape de seuillage
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # trouver les contours dans la cellule seuillée
    cnts: list[np.array] = find_contours(thresh.copy())

    # si aucun contour n'a été trouvé, il s'agit d'une cellule vide
    if not cnts:
        return None

    # Trouver le contour avec la plus grande surface et l'appliquer un masque
    mask: np.array = get_mask(cnts, thresh.shape)

    # calculer le pourcentage de pixels masqués par rapport à la surface totale de l'image
    percentFilled: float = get_filled_percentage(thresh, mask)

    # si moins de 3% du masque est rempli, nous allons considérer qu'il n'y a que du bruit
    if percentFilled < 0.03:
        return None
    # appliquer le masque à la cellule seuillée
    digit: np.array = cv2.bitwise_and(thresh, thresh, mask=mask)

    # vérifier si nous devons visualiser l'étape de masquage
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    return digit


def find_cell_location(warped: np.array, model, debug: bool = False):
    board = np.zeros((9, 9), dtype="int")
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    for y in range(9):
        # initialize the current list of cell locations
        row = []
        startY = y * stepY
        endY = (y + 1) * stepY
        for x in range(9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            endX = (x + 1) * stepX
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=debug > 0)
            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred
        # add the row to our cell locations
        cellLocs.append(row)
    return cellLocs, board
