# import the necessary packages
from pyimagesearch.Sudoku.puzzle import find_cell_location
from pyimagesearch.Sudoku.puzzle import find_puzzle
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

from pyimagesearch.Sudoku.solver import solver
from pyimagesearch.utilities.print_answer import print_answer
from utilities.argparse import parse_arg
from utilities.load_image import load_image


def main():  # sourcery skip: raise-specific-error
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    args = parse_arg(ap)

    print("[INFO] loading digit classifier...")
    model = load_model(args["model"])

    print("[INFO] processing image...")
    image: np.array = load_image(args["image"])

    print("[INFO] find puzzle...")
    puzzleImage, warped = find_puzzle(image, debug=args["debug"] > 0)

    print("[INFO] find cell location...")
    cellLocs, board = find_cell_location(warped, model, debug=args["debug"] > 0)

    if solution := solver(board.tolist()):
        puzzleImage = print_answer(cellLocs, solution, puzzleImage)
        cv2.imshow("Sudoku Result", puzzleImage)
        cv2.waitKey(0)
    else:
        raise Exception("No solution found")


if __name__ == "__main__":
    main()
