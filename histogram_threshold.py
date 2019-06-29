from utils.preprocessing_utils import *
import constants
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", "-d", help = "Directory where the dataset is stored", required=False, default=constants.PATCH_OUTPUT_DIRECTORY)
    parser.add_argument("--black_white_threshold", "-bw", type=float, required=False, default=constants.BLACK_WHITE_THRESHOLD)
    parser.add_argument("--remove_threshold", "-t", type=float, required=False, default=constants.REMOVE_THRESHOLD)
    parser.add_argument("--blur_radius", "-r", type=int, required=False, default=constants.BLUR_RADIUS)
    parser.add_argument("--export", "-e", action="store_true", default=constants.EXPORT_RESULTS)
    parser.add_argument("--out_dir", "-o", required="--export" in sys.argv, default=constants.EXPORT_DIR)

    args = parser.parse_args()

    if os.path.isdir(args.dir):
        if args.export and args.out_dir is None:
            args.out_dir = args.dir

        apply_histogram_thresholding(args.dir, args.black_white_threshold, args.remove_threshold, args.blur_radius, args.export, args.out_dir)
