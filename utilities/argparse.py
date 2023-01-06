def parse_arg(ap):
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained digit classifier")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input Sudoku puzzle image")
    ap.add_argument("-d", "--debug", type=int, default=-1,
                    help="whether we are visualizing each step of the pipeline")
    return vars(ap.parse_args())