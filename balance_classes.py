from utils.preprocessing_utils import *
import argparse
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--accept_margin", "-a", type=int, default=1000)
    parser.add_argument("--max_overlap", "-m", type=int, default=64)
    parser.add_argument("--min_overlap_difference", "-d", type=int, default=8)

    args = parser.parse_args()

    balance_classes_overlap(constants.PATCH_OUTPUT_DIRECTORY,
        args.accept_margin,args.max_overlap, args.min_overlap_difference)
