import os
import sys
import shutil

import constants
from patch import Patch
from construct_dataset import load_slide, get_patch_generator, threshold_image

def get_patches_for_slide(slide_path,
                            file_extension=constants.SLIDE_FILE_EXTENSION,
                            output_dir=constants.PATCH_EVAL_DIRECTORY):
    file_name = os.path.basename(slide_path)

    if not os.path.isfile(slide_path) or not file_name.endswith(file_extension):
        print(f"No valid slide file found at {slide_path}")
        return

    patch_name_to_coords_map = {}

    slide = load_slide(slide_path)
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_dir = os.path.join(output_dir, slide_name)

    if os.path.isdir(slide_dir):
        shutil.rmtree(slide_dir)

    os.makedirs(slide_dir)

    tiles = get_patch_generator(slide)
    tile_size = constants.PATCH_SIZE + (2 * constants.OVERLAP)
    level = len(tiles.level_tiles) - 1
    x_tiles, y_tiles = tiles.level_tiles[level] #Note: Highest level == Highest resolution
    tiled_dims = (y_tiles, x_tiles)
    print(f"Tiling Dimensions: {tiled_dims}")


    x, y = 0, 0
    patch_counter = 0

    patch_name_list = []
    coordinate_list = []

    while y < y_tiles:
        while x < x_tiles:
            patch_coords = tiles.get_tile_coordinates(level, (x,y))
            patch = Patch(patch_coords)
            patch_coordinates = (y,x)
            coordinate_list.append(patch_coordinates)

            patch.img = np.array(tiles.get_tile(level, (x,y)), dtype=np.uint8)
            if np.shape(patch.img) == (tile_size, tile_size, 3):
                if not threshold_image(patch.img, remove_threshold=constants.DEFAULT_CLASS_REMOVE_THRESHOLD) and constants.HISTOGRAM_THRESHOLD:
                    patch_name = os.path.join(slide_dir,
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


    return patch_name_to_coords_map, tiled_dims


if __name__ == "__main__":
    TEST_SLIDE = "/dp/datasets/FL/raw_slides/slide_imgs/FLN01/Scan1/FLN01_Scan1.qptiff"
