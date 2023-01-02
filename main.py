import cv2 as cv

from pyimagesearch.Sudoku.puzzle import find_puzzle

find_puzzle(cv.imread('puzzle.jpg'), debug=True)