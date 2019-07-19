import os
import sys
import re
import csv

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import shutil

import pickle
import constants
import openslide

from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
from matplotlib.path import Path
from patch import Patch
from utils.file_utils import write_pickle_to_disk, load_pickle_from_disk

from collections import defaultdict
from random import shuffle



def construct_training_dataset(slide_file_directory,
        file_extension,
        output_dir,
        annotation_csv_directory):
    """
    Recursively searches for files of the given slide file format starting at
    the provided top level directory.  As slide files are found, they are broken
    up into nonoverlapping patches that can be used to train our model

    Args:
        slide_file_directory (String): Location of the top-level directory, within which
                                      lie all of our files
        file_extension (String): File extension for slide files
        output_dir (String): Folder in which patch files will be saved
        annotation_csv_directory (String): Path to top level directory containing slide annotation csv files
        annotations_only (Boolean): When true, only saves patches that have at least one corner within an annotation path
    Returns:
        None (Patches saved to disk)
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    annotation_class_dirs, output_class_dirs = create_directory_structure()

    slide_name_to_tile_dims_map = {}
    patch_name_to_coords_map = {}

    for root, dirnames, filenames in os.walk(slide_file_directory):
        for filename in filenames:
            if filename.endswith(file_extension) and filename not in constants.FILES_TO_SKIP:
            # if filename == "FLN06_Scan1.qptiff":

                full_path = os.path.join(root, filename)
                slide_name = os.path.splitext(os.path.basename(full_path))[0]

                print("Splitting " + slide_name)

                slide_class_dirs = {class_name:os.path.join(dir, slide_name)
                    for class_name, dir in output_class_dirs.items()}


                for class_dir in slide_class_dirs.values():
                    os.makedirs(class_dir)

                slide = load_slide(full_path)

                class_path_lists = {class_name:construct_annotation_path_list(slide_name, dir)
                    for class_name, dir in annotation_class_dirs.items()}


                tiles = get_patch_generator(slide)
                tile_size = constants.PATCH_SIZE + (2 * constants.OVERLAP)
                level = len(tiles.level_tiles) - 1
                x_tiles, y_tiles = tiles.level_tiles[level] #Note: Highest level == Highest resolution
                tiled_dims = (y_tiles, x_tiles)
                print(f"Tiling Dimensions: {tiled_dims}")
                slide_name_to_tile_dims_map[slide_name] = tiled_dims

                x, y = 0, 0
                patch_counter = 0
                num_since_annotation = 0
                default_counter = 0

                patch_name_list = []
                coordinate_list = []

                while y < y_tiles:
                    while x < x_tiles:
                        patch_coords = tiles.get_tile_coordinates(level, (x,y))
                        patch = Patch(patch_coords)
                        patch_coordinates = (y,x)
                        coordinate_list.append(patch_coordinates)

                        for class_name, path_list in class_path_lists.items():
                            if patch_in_paths(patch, path_list):
                                num_since_annotation = 0
                                patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
                                if np.shape(patch.img) == (tile_size, tile_size, 3):
                                    if not threshold_image(patch.img, remove_threshold=constants.REMOVE_THRESHOLD) and constants.HISTOGRAM_THRESHOLD:
                                        patch_name = os.path.join(slide_class_dirs[class_name], slide_name + "_" + str(patch_counter))
                                        patch_name_list.append(patch_name)
                                        patch.save_img_to_disk(patch_name)
                                        patch_name_to_coords_map[patch_name] = patch_coordinates
                                        patch_counter += 1
                                        if patch_counter % 1000 == 0:
                                            print(patch_counter)
                                        break
                        else:
                            default_counter += 1
                            num_since_annotation += 1
                            if num_since_annotation < 15 or default_counter % 200 == 0:
                                patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
                                if np.shape(patch.img) == (tile_size, tile_size, 3):
                                    if not threshold_image(patch.img, remove_threshold=constants.DEFAULT_CLASS_REMOVE_THRESHOLD) and constants.HISTOGRAM_THRESHOLD:
                                        patch_name = os.path.join(slide_class_dirs[constants.DEFAULT_CLASS_NAME],
                                            slide_name + "_" + str(patch_counter))
                                        patch_name_list.append(patch_name)
                                        patch.save_img_to_disk(patch_name)
                                        patch_name_to_coords_map[patch_name] = patch_coordinates
                                        patch_counter += 1
                                        if patch_counter % 1000 == 0:
                                            print(patch_counter)

                        x += 1
                    y += 1
                    x = 0

                print("Total patches for " + slide_name + ": " + str(patch_counter))

    if os.path.exists(constants.VISUALIZATION_HELPER_FILE_FOLDER):
        shutil.rmtree(constants.VISUALIZATION_HELPER_FILE_FOLDER)

    os.makedirs(constants.VISUALIZATION_HELPER_FILE_FOLDER)

    write_pickle_to_disk(constants.PATCH_NAME_TO_COORDS_MAP, patch_name_to_coords_map)
    write_pickle_to_disk(constants.SLIDE_NAME_TO_TILE_DIMS_MAP, slide_name_to_tile_dims_map)

def threshold_image(image, bw_threshold=constants.BLACK_WHITE_THRESHOLD, blur_radius=constants.BLUR_RADIUS,
    remove_threshold=constants.REMOVE_THRESHOLD):

    im = Image.fromarray(image)
    # convert to grayscale
    im = im.convert('L')
    # add gaussian blur
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_vals = np.array(im) / 255
    percent_whitespace = np.mean((pixel_vals > bw_threshold).astype(np.uint8))

    return percent_whitespace > remove_threshold

def create_directory_structure():
    if not os.path.isdir(constants.HELPER_FILES_DIRECTORY):
        os.makedirs(constants.HELPER_FILES_DIRECTORY)

    annotation_class_dirs = {}
    output_class_dirs = {}

    for annotation_class_dir in os.listdir(constants.ANNOTATION_CSV_DIRECTORY):
        class_name = annotation_class_dir.replace("_csv_files", "")
        annotation_class_dirs[class_name] = os.path.join(constants.ANNOTATION_CSV_DIRECTORY, annotation_class_dir)

        output_class_dir = os.path.join(constants.PATCH_OUTPUT_DIRECTORY, class_name)
        if not os.path.isdir(output_class_dir):
            os.makedirs(output_class_dir)

        output_class_dirs[class_name] = output_class_dir

    default_class_dir = os.path.join(constants.PATCH_OUTPUT_DIRECTORY, constants.DEFAULT_CLASS_NAME)
    if not os.path.isdir(default_class_dir):
        os.makedirs(default_class_dir)

    output_class_dirs[constants.DEFAULT_CLASS_NAME] = default_class_dir

    write_pickle_to_disk(os.path.join(constants.HELPER_FILES_DIRECTORY, "annotation_class_dirs"), annotation_class_dirs)
    write_pickle_to_disk(os.path.join(constants.HELPER_FILES_DIRECTORY, "output_class_dirs"), output_class_dirs)

    return annotation_class_dirs, output_class_dirs


def load_slide(path):
    """
    Function for opening slide images

    Args:
        path: Path to the image file

    Returns:
        OpenSlide object

    """

    osr = openslide.OpenSlide(path)
    return osr

def get_slide_thumbnail(path, height, width):
    """
    Returns a thumbnail of the slide found at path

    Args
        path (String): Path to slide file
    Returns:
        thumbnail (PIL Image): Image object
    """
    osr = openslide.OpenSlide(path)
    thumbnail = osr.get_thumbnail((height, width))
    return thumbnail

def get_patch_generator(slide, tile_size=constants.PATCH_SIZE, overlap=constants.OVERLAP, limit_bounds=False):
    """
    Returns a generator that splits an OpenSlide object into patches

    Args:
        slide: OpenSlide object
        tile_size: Width and height of a single tile
        overlap: Number of extra pixels to add to each interior edge of a tile
        limit_bounds: If True, renders only non-empty slide region
    Returns:
        DeepZoomGenerator
    """
    return DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap,
                limit_bounds=limit_bounds)



def construct_annotation_path_list(slide_name, annotation_base_path):
    """
    Given the name of a slide, returns a list of polygons representing the annotations
    drawn on that slide.

    Args:
        slide_name (String): Name of scanned slide
        annotation_base_path (String): Path to top level directory containing slide annotations
    Returns:
        path_list (Path list): List of Path objects representing the annotations on the given slide
    """

    full_annotation_dir = os.path.join(annotation_base_path, slide_name)
    if not os.path.exists(full_annotation_dir):
        return []

    annotation_list = []

    for filename in os.listdir(full_annotation_dir):
        if filename.endswith(".csv"):
            annotation_file = os.path.join(full_annotation_dir, filename)
            current_annotation = read_annotation(annotation_file)
            annotation_list.append(current_annotation)

    path_list = list(map(construct_annotation_path, annotation_list))
    return path_list

def read_annotation(csv_path):
    """
    Loads the coordinates of an annotation created with QuPath
    and stored in a csv file

    Args:
        csv_path (str): Path to csv file containing annotation

    Returns:
        vertex_list (Nx2 numpy array): Nx2 array containing all
                                       vertices in the annotaiton
    """
    df = pd.read_csv(csv_path, header=None)

    return df.values

def construct_annotation_path(vertices):
    """
    Constructs a matplotlib Path object that represents the polygon
    with provided vertices

    Args:
        vertices (Nx2 numpy array): vertices of our polygon

    Returns:
        path (Path object): Path object representing our polygon

    """
    polygon = Path(vertices)
    return polygon

def patch_in_paths(patch, path_list):
    """
    Utility function to check if a given patch object is contained within
    any of the annotation paths in path_list

    Args:
        patch (Patch): Patch object that we want to check
        path_list (Path list): List of annotation paths for a slide
    Returns:
        in_path (Boolean): True if patch contained within one of the paths in path_list
    """

    in_path = False
    for path in path_list:
        if patch.vertices_in_annotation(path, constants.NUM_VERTICES_IN_ANNOTATION):
            in_path = True

    return in_path

if __name__ == "__main__":
    print("Building dataset...")
    construct_training_dataset(
        constants.SLIDE_FILE_DIRECTORY,
        constants.SLIDE_FILE_EXTENSION,
        constants.PATCH_OUTPUT_DIRECTORY,
        constants.ANNOTATION_CSV_DIRECTORY
    )

    print("Training dataset successfully constructed!")
