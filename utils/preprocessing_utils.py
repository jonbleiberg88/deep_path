import constants
import os
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def map_slide_to_bw(path, threshold=0.9, blur_radius=7):
    im = Image.open(path)
    # convert to grayscale
    im = im.convert('L')
    # add gaussian blur
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_vals = np.array(im) / 255

    return (pixel_vals > threshold).astype(np.uint8)

def get_percent_whitespace(data_dir, threshold=0.9, blur_radius=7):
    whitespace = []
    idx = 0
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                idx += 1
                if idx % 2000 == 0:
                    print(f"{idx} files processed...")
                full_path = os.path.join(root, filename)
                whitespace.append((full_path, np.mean(map_slide_to_bw(full_path, threshold, blur_radius))))

    return whitespace

def threshold(whitespace, threshold = 0.9):
    initial_count = len(whitespace)
    remove_count = 0
    for file, percent in whitespace:
        if percent > threshold:
            whitespace.remove((file, percent))
            os.remove(file)
            remove_count += 1

    print(f'{remove_count} of {initial_count} files removed')

    return whitespace

def whitespace_to_csv(whitespace, out_dir):
    df = pd.DataFrame(whitespace, columns=['file_path', 'percent_whitespace'])
    df.to_csv(f'{out_dir}/percent_whitespace.csv')


def apply_histogram_thresholding(data_dir, bw_threshold=0.9, remove_threshold=0.9,
        blur_radius=7, export=True, export_dir=None):
    if export and export_dir is None:
        print("Please specify an export directory for the output csv file")
        return

    print("Calculating white space percentages...")
    whitespace = get_percent_whitespace(data_dir, bw_threshold, blur_radius)

    print("Removing thresholded files...")
    whitespace = threshold(whitespace, remove_threshold)

    if export:
        print("Exporting to csv...")
        whitespace_to_csv(whitespace, export_dir)

def get_histogram_for_img(path, blur_radius=7):
    im = Image.open(path)
    # convert to grayscale
    im = im.convert('L')
    # add gaussian blur
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_vals = np.array(im) / 255
    plt.hist(pixel_vals.reshape(-1))

    img_name = os.path.basename(path)

    plt.title(f"Pixel Intensity Histogram for {img_name}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")
    plt.show(block=True)

def get_histogram_for_dir(data_dir, bw_threshold=0.9, blur_radius=7):
    whitespace = get_percent_whitespace(data_dir, bw_threshold, blur_radius)
    white_vals = np.array([val for _,val in whitespace])

    dir_name = os.path.basename(data_dir[:-1])

    plt.hist(white_vals)
    plt.title(f"Percent White Space Histogram for {dir_name}")
    plt.xlabel("% White Space")
    plt.ylabel("# of images")

    plt.show()
