from utils.preprocessing_utils import *
import argparse
import os

path = "/Volumes/Elements/5:21:19/Datasets/FL/256_0_4/small_tumor_cells/FLT13_Scan1/FLT13_Scan1_64.jpg"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path","-p", type=str, help="Path to image/directory to visualize", required=True)
    args = parser.parse_args()

    if os.path.isfile(args.path):
        get_histogram_for_img(args.path)
    elif os.path.isdir(args.path):
        get_histogram_for_dir(args.path)
    else:
        print("Invalid file path")
