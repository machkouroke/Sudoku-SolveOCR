# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    # convertir l'image en niveaux de gris et la flouter légèrement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # appliquer un seuillage adaptatif, puis inverser la carte de seuil
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    # vérifier si nous visualisons chaque étape de l'image
    # pipeline de traitement (dans ce cas, seuillage)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # trouver des contours dans l'image seuillée et les trier par taille dans l'ordre décroissant
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # initialiser un contour qui correspond au contour du puzzle
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
    # si le contour du puzzle est vide, alors notre script n'a pas pu trouver le contour du puzzle Sudoku donc générer
    # une erreur
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))
    # vérifier si nous visualisons le contour du puzzle Sudoku détecté
    if debug:
        # dessinez le contour du puzzle sur l'image, puis affichez-le sur notre
        # écran à des fins de visualisation/débogage
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    # appliquez une transformation de perspective à quatre points à la fois à l'image d'origine
    # et à l'image en niveaux de gris pour obtenir une vue plongeante du puzzle de haut en bas
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # vérifier si nous visualisons la transformation de perspective
    if debug:
        # afficher l'image de sortie déformée (encore une fois, à des fins de débogage)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    # renvoie un 2-tuple de puzzle en RVB et en niveaux de gris
    return puzzle, warped


def extract_digit(cell, debug=False):
    """
    For a g
    :param cell:
    :param debug:
    :return:
    """
    # appliquer un seuillage automatique à la cellule, puis effacer toutes les bordures connectées
    # qui touchent la bordure de la cellule
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # vérifier si nous visualisons l'étape de seuillage cellulaire
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    # trouver les contours dans la cellule seuillée
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # si aucun contour n'a été trouvé, il s'agit d'une cellule vide
    if len(cnts) == 0:
        return None
    # sinon, trouvez le plus grand contour dans la cellule et
    # créez un masque pour le contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    # calculer le pourcentage de pixels masqués par rapport à
    # la surface totale de l'image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # si moins de 3 % du masque est rempli, nous examinons le bruit et
    # pouvons ignorer le contour en toute sécurité
    if percentFilled < 0.03:
        return None
    # appliquer le masque à la cellule seuillée
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # vérifier si nous devons visualiser l'étape de masquage
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # renvoie le chiffre à la fonction appelante
    return digit
