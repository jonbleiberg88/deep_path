import constants
import os
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shutil
import pickle
import openslide
from openslide.deepzoom import DeepZoomGenerator
from matplotlib.path import Path
from patch import Patch
from utils.file_utils import write_pickle_to_disk
from utils.slide_utils import *
import os
import sys
import re
from collections import defaultdict
from random import shuffle



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
            os.remove(file)
            remove_count += 1

    print(f'{remove_count} of {initial_count} files removed')

    return whitespace

def whitespace_to_csv(whitespace, out_dir):
    df = pd.DataFrame(whitespace, columns=['file_path', 'percent_whitespace'])
    df.to_csv(f'{out_dir}/percent_whitespace.csv', index=False)


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

def run_overlap_augmentation(data_dir, max_images=10e5, max_overlap=64, min_overlap_diff=8):
    patch_counts = defaultdict(int)
    aug_patch_to_coords = {}
    aug_slide_to_dims = {}

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        for img_dir in os.listdir(class_path):
            img_path = os.path.join(class_path, img_dir)
            max_idx = 0
            for img_name in os.listdir(img_path):
                if "_aug.jpg" in img_name:
                    idx = int(img_name.split("_")[-2])
                else:
                    idx = int(img_name.rpartition("_")[-1][:-4])
                max_idx = max(max_idx, idx)
            patch_counts[img_dir] = max_idx

    prev_image_count = sum(list(patch_counts.values()))

    aug_round = 1
    if max_overlap >= constants.INPUT_IMAGE_DIM[0]:
        max_overlap = constants.INPUT_IMAGE_DIM[0] -1
    overlap_vals = np.append(np.arange(0,constants.OVERLAP, min_overlap_diff),
        np.arange(constants.OVERLAP, max_overlap+1, min_overlap_diff)[1:])

    total_image_count = prev_image_count

    if total_image_count > max_images:
        print(f"Current image count ({total_image_count}) already exceeds the given maximum of {max_images}!")
        return

    while max_images > total_image_count:

        if len(overlap_vals) == 0:
            print("All possible overlap values exhausted")
            return

        idx = np.random.randint(0, len(overlap_vals))
        new_overlap = overlap_vals[idx]
        overlap_vals = np.delete(overlap_vals,idx)

        new_tile_size = constants.INPUT_IMAGE_DIM[0] - (new_overlap * 2)

        print(f"Beginning augmentation round {aug_round} with overlap {new_overlap}")
        aug_count, aug_small, aug_large, patch_counts, round_patch_to_coords, round_slide_to_dims= augment_all_classes(new_tile_size, new_overlap, total_image_count, aug_round,
            patch_counts, total_image_count, max_images)
        total_image_count += aug_count

        aug_patch_to_coords[aug_round] = round_patch_to_coords
        aug_slide_to_dims[aug_round] = round_slide_to_dims

        print(f"Added {aug_small} small patches and {aug_large} large patches {aug_count} patches")
        print( f"Total patches added in round {aug_round}: {aug_count}")

        aug_round += 1

    write_pickle_to_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, "aug_patch_name_to_coords_map"), aug_patch_to_coords)
    write_pickle_to_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, "aug_slide_to_dims_map"), aug_slide_to_dims)


def augment_all_classes(tile_size, overlap,
        img_count, aug_round, patch_counts, current_count, max_images,
        slide_file_dir=constants.SLIDE_FILE_DIRECTORY,
        file_extension = constants.SLIDE_FILE_EXTENSION,
        annotation_csv_directory=constants.ANNOTATION_CSV_DIRECTORY):

    slide_name_to_tile_dims_map = {}
    patch_name_to_coords_map = {}

    total_count = 0
    large_count, small_count = 0, 0
    for root, dirnames, filenames in os.walk(slide_file_dir):
        for filename in filenames:
            if filename.endswith(file_extension) and filename not in constants.FILES_TO_SKIP:

                full_path = os.path.join(root, filename)
                slide_name = os.path.splitext(os.path.basename(full_path))[0]

                slide_large_cells_dir = os.path.join(constants.LARGE_CELL_PATCHES, slide_name)
                slide_small_cells_dir = os.path.join(constants.SMALL_CELL_PATCHES, slide_name)

                slide = load_slide(full_path)

                large_path_list, small_path_list = [], []

                large_path_list = construct_annotation_path_list(slide_name, os.path.join(annotation_csv_directory,
                        "large_tumor_csv_files"))
                small_path_list = construct_annotation_path_list(slide_name, os.path.join(annotation_csv_directory,
                        "small_tumor_csv_files"))

                if large_path_list == [] and small_path_list == []:
                    continue

                print("Splitting " + slide_name)

                tiles = get_patch_generator(slide, tile_size, overlap)
                img_size = tile_size + (2 * overlap)

                level = len(tiles.level_tiles) - 1
                x_tiles, y_tiles = tiles.level_tiles[level] #Note: Highest level == Highest resolution
                tiled_dims = (y_tiles, x_tiles)
                slide_name_to_tile_dims_map[slide_name] = tiled_dims

                x, y = 0, 0
                patch_name_list = []
                coordinate_list = []
                slide_large_count = 0
                slide_small_count = 0

                while y < y_tiles:
                    while x < x_tiles:
                        patch_coords = tiles.get_tile_coordinates(level, (x,y))
                        patch = Patch(patch_coords)
                        patch_coordinates = (y,x)
                        coordinate_list.append(patch_coordinates)

                        if patch_in_paths(patch, small_path_list):
                            patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
                            if np.shape(patch.img) == (img_size, img_size, 3):
                                patch_name = os.path.join(slide_small_cells_dir, f'{slide_name}_{str(patch_counts[slide_name] + 1)}_aug')
                                patch_counts[slide_name] += 1
                                patch_name_list.append(patch_name)
                                patch.save_img_to_disk(patch_name)
                                patch_name_to_coords_map[patch_name] = patch_coordinates
                                slide_small_count += 1
                        elif patch_in_paths(patch, large_path_list):
                            patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
                            if np.shape(patch.img) == (img_size, img_size, 3):
                                patch_name = os.path.join(slide_large_cells_dir, f'{slide_name}_{str(patch_counts[slide_name] + 1)}_aug')
                                patch_counts[slide_name] += 1
                                patch_name_list.append(patch_name)
                                patch.save_img_to_disk(patch_name)
                                patch_name_to_coords_map[patch_name] = patch_coordinates
                                slide_large_count += 1

                        x += 1
                    y += 1
                    x = 0

                total_count += slide_small_count + slide_large_count
                large_count += slide_large_count
                small_count += slide_small_count


                print(f"Total patches for {slide_name}: {slide_large_count + slide_small_count}")
                print(f"Large: {slide_large_count};  Small: {slide_small_count}")

                if total_count + current_count >= max_images:
                    print("Maximum number of images reached!")
                    return

    return total_count, small_count, large_count, patch_counts, patch_name_to_coords_map, slide_name_to_tile_dims_map



def balance_classes_overlap(data_dir, accept_margin = 0.1, max_overlap=64, min_overlap_diff=8):
    """
    Given the top level directory pointing to where our image class folders live,
    attempts to balance the number of images belonging to each class

    Args:
        data_dir (String): Path to top-level directory of dataset
        accept_margin (int or float): Acceptable margin of class Imbalance
        max_overlap (int): Maximum overlap value to try
        min_overlap_diff (int): Minimum space between overlap values
    Returns:
        None (output saved to disk)
    """
    aug_patch_to_coords = {}
    aug_slide_to_dims = {}

    class_dirs = [constants.LARGE_CELL_PATCHES, constants.SMALL_CELL_PATCHES]

    class_counts = {}
    for dir in class_dirs:
        class_count = 0
        for patient in os.listdir(dir):
            full_patient_dir = os.path.join(dir, patient)
            class_count += len(os.listdir(full_patient_dir))
        class_counts[dir] = class_count



    max_class = max(class_counts.items(), key = lambda x: x[1])[0]
    print
    if max_class == constants.SMALL_CELL_PATCHES:
        augment_large = True
        aug_dir = constants.LARGE_CELL_PATCHES
        print("Augmenting large cell patches...")
    else:
        augment_large = False
        aug_dir = constants.SMALL_CELL_PATCHES
        print("Augmenting small cell patches...")

    diff = abs(class_counts[constants.LARGE_CELL_PATCHES] - class_counts[constants.SMALL_CELL_PATCHES])
    print(f"Imbalance of {diff} images...")

    aug_round = 1
    if max_overlap >= constants.INPUT_IMAGE_DIM[0]:
        max_overlap = constants.INPUT_IMAGE_DIM[0] -1
    overlap_vals = np.append(np.arange(0,constants.OVERLAP, min_overlap_diff),
        np.arange(constants.OVERLAP, max_overlap+1, min_overlap_diff)[1:])

    if type(accept_margin) is float:
        accept_margin = int(max(class_counts.values()) * accept_margin)

    patch_counts = defaultdict(int)

    for img_dir in os.listdir(aug_dir):
        img_path = os.path.join(aug_dir, img_dir)
        max_idx = 0
        for img_name in os.listdir(img_path):
            if "_aug.jpg" in img_name:
                idx = int(img_name.split("_")[-2])
            else:
                idx = int(img_name.rpartition("_")[-1][:-4])
            max_idx = max(max_idx, idx)
        patch_counts[img_dir] = max_idx

    while diff > accept_margin:

        if len(overlap_vals) == 0:
            print("Out of suitable overlap values, try increasing the maximum allowed overlap or the minimum overlap difference...")
            return

        idx = np.random.randint(0, len(overlap_vals))
        new_overlap = overlap_vals[idx]
        overlap_vals = np.delete(overlap_vals,idx)

        new_tile_size = constants.INPUT_IMAGE_DIM[0] - (new_overlap * 2)

        print(f"Beginning augmentation round {aug_round} with overlap {new_overlap}")
        aug_count, patch_counts, round_patch_to_coords, round_slide_to_dims = augment_class(aug_dir, augment_large, new_tile_size, new_overlap, diff, aug_round,
            patch_counts, accept_margin)
        diff -= aug_count

        aug_patch_to_coords[aug_round] = round_patch_to_coords
        aug_slide_to_dims[aug_round] = round_slide_to_dims

        print(f"Added {aug_count} patches in round {aug_round}, imbalance is now {diff} patches.")

        aug_round += 1

    write_pickle_to_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, "bal_patch_name_to_coords_map"), aug_patch_to_coords)
    write_pickle_to_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, "bal_slide_to_dims_map"), aug_slide_to_dims)

    print("Class imbalance within acceptable margin!")



def augment_class(class_dir, augment_large, tile_size, overlap, diff, aug_round, patch_counts,
        accept_margin=1000,
        file_extension = constants.SLIDE_FILE_EXTENSION,
        slide_file_dir=constants.SLIDE_FILE_DIRECTORY,
        annotation_csv_directory=constants.ANNOTATION_CSV_DIRECTORY):

    slide_name_to_tile_dims_map = {}
    patch_name_to_coords_map = {}

    total_count = 0
    for root, dirnames, filenames in os.walk(slide_file_dir):
        shuffle(filenames)
        for filename in filenames:
            if filename.endswith(file_extension) and filename not in constants.FILES_TO_SKIP:

                full_path = os.path.join(root, filename)
                slide_name = os.path.splitext(os.path.basename(full_path))[0]

                slide_dir = os.path.join(class_dir, slide_name)

                slide = load_slide(full_path)

                path_list = []

                if augment_large:
                    path_list = construct_annotation_path_list(slide_name, os.path.join(annotation_csv_directory,
                        "large_tumor_csv_files"))
                else:
                    path_list = construct_annotation_path_list(slide_name, os.path.join(annotation_csv_directory,
                        "small_tumor_csv_files"))

                if path_list == []:
                    continue

                print("Splitting " + slide_name)

                tiles = get_patch_generator(slide, tile_size, overlap)
                img_size = tile_size + (2 * overlap)

                level = len(tiles.level_tiles) - 1
                x_tiles, y_tiles = tiles.level_tiles[level] #Note: Highest level == Highest resolution
                tiled_dims = (y_tiles, x_tiles)
                slide_name_to_tile_dims_map[slide_name] = tiled_dims

                x, y = 0, 0
                slide_count = 0
                patch_name_list = []
                coordinate_list = []

                while y < y_tiles:
                    while x < x_tiles:
                        patch_coords = tiles.get_tile_coordinates(level, (x,y))
                        patch = Patch(patch_coords)
                        patch_coordinates = (y,x)
                        coordinate_list.append(patch_coordinates)

                        if patch_in_paths(patch, path_list):
                            patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
                            if np.shape(patch.img) == (img_size, img_size, 3):
                                patch_name = os.path.join(slide_dir, f'{slide_name}_{str(patch_counts[slide_name] + 1)}_aug')
                                patch_counts[slide_name] += 1
                                patch_name_list.append(patch_name)
                                patch.save_img_to_disk(patch_name)
                                patch_name_to_coords_map[patch_name] = patch_coordinates
                                total_count += 1
                                slide_count += 1

                        if total_count > diff - accept_margin:
                            break
                        x += 1
                    if total_count > diff - accept_margin:
                        break
                    y += 1
                    x = 0

                print("Total patches for " + slide_name + ": " + str(slide_count))

    return total_count, patch_counts, patch_name_to_coords_map, slide_name_to_tile_dims_map

def fix_filenames(data_dir):
    max_before_aug = defaultdict(int)

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        for img_dir in os.listdir(class_path):
            img_path = os.path.join(class_path, img_dir)
            max_idx = 0
            for img_name in os.listdir(img_path):
                if "aug" not in img_name:
                    idx = int(img_name.rpartition("_")[-1][:-4])
                    max_idx = max(max_idx, idx)
            max_before_aug[img_dir] = max_idx

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        for img_dir in os.listdir(class_path):
            img_path = os.path.join(class_path, img_dir)
            for img_name in os.listdir(img_path):
                if "aug" in img_name:
                    old_path = os.path.join(img_path, img_name)

                    curr_idx = max_before_aug[img_dir] + 1
                    new_name = "_".join(img_name.split("_")[:-2]) + "_" + str(curr_idx) + ".jpg"
                    new_path = os.path.join(img_path, new_name)
                    os.rename(old_path, new_path)

                    max_before_aug[img_dir] = curr_idx
