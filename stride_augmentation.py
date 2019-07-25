from utils.preprocessing_utils import run_overlap_augmentation
import argparse
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--max_images", "-n", type=int, default=constants.MAX_IMAGES)
    parser.add_argument("--max_overlap", "-m", type=int, default=constants.MAX_OVERLAP)
    parser.add_argument("--min_overlap_difference", "-d", type=int, default=constants.MIN_OVERLAP_DIFFERENCE)

    args = parser.parse_args()

    run_overlap_augmentation(constants.PATCH_OUTPUT_DIRECTORY, args.max_images, args.max_overlap,
        args.min_overlap_difference)
