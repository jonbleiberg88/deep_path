from utils.preprocessing_utils import balance_classes_overlap
import argparse
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--accept_margin", "-a", type=int)
    parser.add_argument("--accept_thresh", "-at",type=float)
    parser.add_argument("--max_overlap", "-m", type=int, default=127)
    parser.add_argument("--min_overlap_difference", "-d", type=int, default=2)

    args = parser.parse_args()

    if args.accept_thresh is not None:
        accept = args.accept_thresh
    elif args.accept_margin is not None:
        accept = args.accept_margin
    else:
        accept = 0.1

    balance_classes_overlap(constants.PATCH_OUTPUT_DIRECTORY,
        accept,args.max_overlap, args.min_overlap_difference)
