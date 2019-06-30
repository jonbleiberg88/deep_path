import os
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import matplotlib
import matplotlib.pyplot as plt

import constants
from utils.file_utils import load_pickle_from_disk


def create_preds_array(slide_name):
    """
    Generates an 2D array with the values predicted by the model in their proper
     relative places on the slide

    Args:
        slide_name (str): the name of the slide to extract predictions from

    Returns:
        (2D numpy array): 2D array containing the predicted class confidence values
        (list): List of coordinates containing a predicted value for computational
            efficiency
    """
    slide_to_dims = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/slide_name_to_tile_dims_map")
    dims = slide_to_dims[slide_name]
    preds_array = np.full(dims, fill_value=np.nan, dtype=np.float32)

    preds_file = f"{constants.PREDICTIONS_DIRECTORY}/{slide_name}.csv"
    df = pd.read_csv(preds_file)
    patch_to_coords = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/patch_name_to_coords_map")
    coords_list = []

    for _, row in df.iterrows():
        coords = patch_to_coords[row['filepath'].replace('.jpg', '')]
        preds_array[coords] = row['prediction']
        coords_list.append(coords)

    return preds_array, coords_list

def knn_smooth(preds_array, coords, knn_range=constants.KNN_RANGE, smooth_factor=constants.SMOOTH_FACTOR):
    """
    Smoothes the predicted class confidence values by taking a weighted average of surrounding
    patches' predicted class values

    Args:
        preds_array (2D numpy array): Predicted class confidence values from create_preds_array
        coords (list): List of coordinates containing a predicted patch
        knn_range (int, optional): Farthest distance of patches to include in the weighted average.
            Defaults to value specified in constants.py.
        smooth_factor (float, optional): Relative weighting of surrounding patches compared to
            the center patch. Defaults to value specified in constants.py.

    Results:
        (2D numpy array): Array containing the smoothed confidence values

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

    Args:
        preds_array (2D numpy array): array of predicted confidences from create_preds_array
        label_to_class (dict): dictionary to convert labels to class names

    Returns:
        (defaultdict): Dict containing the number of patches predicted for a given class
        (defaultdict): Dict containing the estimated surface area for a given class
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

    Args:
        slide_name (str): name of the slide to process

    Returns:
        (defaultdict): Default dictionary containing the surface area values for a
            given class
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

def visualize_predictions(preds_array, slide, label_to_class, mode='save'):
    """
    Visualizes class label predictions for a given slide, either printed to
    jupyter notebook or saved in a file in the visualization helper files folder

    Args:
        preds_array (numpy array): Array of predicted confidences from create_preds_array
        slide (str): name of slide to visualize
        label_to_class (dict): dictionary to convert labels to class names
        mode (str, optional): Either 'save', which saves the visualization to disk or
            'jupyter' which can show the visualization in an IPython console. Defaults
            to 'save'.

    Returns:
        None (output printed or saved to disk based on mode setting)

    """
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='red')


    plt.imshow(preds_array, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(f"{slide} Classification Confidences")
    cbar = plt.colorbar(ticks=[0.0, 0.5, 1.0])
    cbar.set_ticklabels([process_label(label_to_class[0]), '', process_label(label_to_class[1])])

    if mode == 'jupyter':
        plt.show()

    elif mode == 'save':
        viz_dir = f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/visualizations"
        if not os.path.isdir(viz_dir):
            os.makedirs(viz_dir)
        plt.savefig(f"{viz_dir}/{slide}.png")

def process_label(label):
    """
    Helper function to format class names for nice printing

    Args:
        label (str): the class label to format

    Returns:
        (str): A formatted version of the class label for printing
    """
    return label.replace("_"," ").title()

def get_metrics(num_per_class, sa_dict, preds_array, class_to_label):
    """
    Calculates the predicted and true surface area ratios for each class

    Args:
        num_per_class (defaultdict): Number of patches predicted for each class.
            Output of estimate_surface_areas.
        sa_dict (defaultdict): True surface areas for each class extrated from quPath.
            Output of get_sa_for_slide.
        preds_array (2D ndarray): Predicted class confidence values for the slide.
            Output of knn_smooth or create_preds_array.
        class_to_label (dict): dictionary to convert class names to labels

    Returns:
        (defaultdict): Dict containing the true and predicted class surface area ratios
            for each class
    """
    total_num = sum(list(num_per_class.values()))

    true_total_sa = sum([v for k,v in sa_dict.items() if k != "normal_marrow"])

    results_dict = defaultdict(lambda: {'true': None, 'pred': None})

    for class_name, val in num_per_class.items():
        true_ratio = sa_dict[class_name] / true_total_sa
        predicted_ratio = val / total_num

        results_dict[class_name]['true'] = true_ratio
        results_dict[class_name]['pred'] = predicted_ratio

        if class_to_label[class_name] == 1:
            mean_confidence = np.nanmean(preds_array)
            print(f"Results for class {process_label(class_name)}:")
            print(f"Predicted ratio: {predicted_ratio:.2f}; True ratio: {true_ratio:.2f}; Mean Confidence: {mean_confidence:.2f}")
        else:
            print(f"Results for class {process_label(class_name)}:")
            print(f"Predicted ratio: {predicted_ratio:.2f}; True ratio: {true_ratio:.2f}")

    return results_dict

def process_predictions(slide):
    """
    Outputs and visualizes the predicted confidences and class labels for a given slide

    Args:
        slide (str): the slide to process

    Returns:
        (defualtdict): Dict containing the true and predicted class surface area ratios
            for each class

    """

    preds_array, coords = create_preds_array(slide)
    if constants.KNN_SMOOTH:
        preds_array = knn_smooth(preds_array, coords)

    class_to_label = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/class_to_label")
    ### DELETE on NEXT RUN
    class_to_label['large_tumor'] = 0
    class_to_label['small_tumor'] = 1
    ###
    label_to_class = {v:k for k,v in class_to_label.items()}

    num_per_class, sa_per_class = estimate_surface_areas(preds_array, label_to_class)

    visualize_predictions(preds_array, slide, label_to_class)

    sa_dict = get_sa_for_slide(slide)
    results_dict = get_metrics(num_per_class, sa_dict, preds_array, class_to_label)
    return results_dict

def process_all_predictions():
    """
    Outputs and visualizes the predicted confidences and class labels for all slides
    in the predictions directory

    Args:
        None

    Returns:
        None
    """
    for slide_file in os.listdir(constants.PREDICTIONS_DIRECTORY):
        slide = slide_file.replace(".csv", "")
        print(f"Results for Slide {slide}")
        process_predictions(slide)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--slide", type=str, required=False, help="Slide for which to output predictions")

    args = parser.parse_args()

    if args.slide is None:
        process_all_predictions()
    else:
        process_predictions(args.slide)
