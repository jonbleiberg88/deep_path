import os
import numpy as np
import pandas as pd
from collections import defaultdict

import constants
from utils.file_utils import load_pickle_from_disk


def create_preds_array(slide_name):
    """
    Generates an 2D array with the values predicted by the model in their proper
     relative places on the slide
    """
    slide_to_dims = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/slide_name_to_tile_dims_map")
    dims = slide_to_dims[slide_name]
    preds_array = np.full(dims, fill_value=np.nan, dtype=np.float32)

    preds_file = f"{constants.PREDICTIONS_DIRECTORY}/{slide_name}.csv"
    df = pd.read_csv(preds_file)
    patch_to_coords = load_pickle_from_disk(f"{vis}/patch_name_to_coords_map")
    coords_list = []

    for _, row in df.iterrows():
        coords = patch_to_coords[row['filepath'].replace('.jpg', '')]
        preds_array[coords] = row['prediction']
        coords_list.append(coords)

    return preds_array, coords_list

def knn_smooth(preds_array, coords, knn_range=1, smooth_factor=0.7):
    """
    Applies KNN smoothing to the 2D predictions array
    """
    result_array = np.copy(preds_array)

    for c in coords:
        x, y = c
        adj = preds_array[x-knn_range:x+knn_range+1, y-knn_range:y+knn_range+1]
        if smooth_factor != 1:
            weights = np.invert(np.isnan(adj)) * smooth_factor
            weights[knn_range, knn_range] = 1
            result_array[c] = np.nansum(adj * weights) / np.sum(weights)
        else:
            result_array[c] = np.nanmean(adj)

    return result_array

def estimate_surface_areas(preds_array, label_to_class):
    """
    Generates rough surface area estimates for a given slide based on patch size
    """
    class_preds = preds_array.round()
    n_classes = np.nanmax(class_preds) + 1
    num_per_class = defaultdict(int)


    for label in range(int(n_classes)):
        num_per_class[label_to_class[label]] = np.sum(class_preds == label)

    patch_area = constants.PATCH_SIZE ** 2
    sa_per_class = {c:n*patch_area for c,n in num_per_class.items()}

    return num_per_class, sa_per_class

def get_sa_for_slide(slide_name):
    """
    Extracts true annotation surface area values by class from quPath stored values
    for a given slide
    """
    path = f"{constants.SA_CSV_DIRECTORY}/{slide_name}.csv"
    sa_df = pd.read_csv(path)
    sa_df.columns = ['class', 'area']
    sa_by_class = sa_df.groupby('class').sum()
    sa_dict = defaultdict(int)

    for class_name, val in sa_by_class.iterrows():
        area = val[0]
        sa_dict[class_name] = area

    return sa_dict

def visualize_predictions(preds_array):
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='red')
    
    plt.imshow(preds_array, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title("Patch Classification Confidences")
    cbar = plt.colorbar(ticks=[0.0, 0.5, 1.0])
    cbar.set_ticklabels([process_label(label_to_class[0]), '', process_label(label_to_class[1])])


    plt.show()

def process_label(label):
    return label.replace("_"," ").title()
