from utils.preprocessing_utils import *
import constants
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", "-d", help = "Directory where the dataset is stored", required=True, default=constants.PATCH_OUTPUT_DIRECTORY)
    parser.add_argument("--black_white_threshold", "-bw", type=float, required=False, default=0.9)
    parser.add_argument("--remove_threshold", "-t", type=float, required=False, default=0.9)
    parser.add_argument("--blur_radius", "-r", type=int, required=False, default=7)
    parser.add_argument("--export", "-e", action="store_true")
    parser.add_argument("--out_dir", "-o", required="--export" in sys.argv)

    args = parser.parse_args()

    if os.path.isdir(args.dir):
        if args.export and args.out_dir is None:
            args.out_dir = args.dir

        apply_histogram_thresholding(args.dir, args.black_white_threshold, args.remove_threshold, args.blur_radius, args.export, args.out_dir)
