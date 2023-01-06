import numpy as np
import cv2


def print_answer(cellLocs: list, solution: list, puzzleImage: np.array) -> np.array:
    for (cellRow, boardRow) in zip(cellLocs, np.array(solution)):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # draw the result digit on the Sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return puzzleImage
