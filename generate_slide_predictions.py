import os
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import openslide

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
        (tuple): (width, height) tuple of slide tile dimensions
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

    return preds_array, coords_list, dims, df


def get_confusion_matrix(slide, preds_df):
    confusion_mat = np.zeros((2, 2), dtype=np.int32)

    for _,row in preds_df.iterrows():
        pred = int(round(row['prediction']))
        confusion_mat[row['labels'], pred] +=1

    return confusion_mat


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    Pretty print for confusion matrices
    from https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}s".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

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
        adj = preds_array[max(x-knn_range, 0):x+knn_range+1, max(y-knn_range,0):y+knn_range+1]
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
    n_classes = 2
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

def visualize_predictions(preds_array, slide, label_to_class, dims, mode='save'):
    """
    Visualizes class label predictions for a given slide, either printed to
    jupyter notebook or saved in a file in the visualization helper files folder

    Args:
        preds_array (numpy array): Array of predicted confidences from create_preds_array
        slide (str): name of slide to visualize
        label_to_class (dict): dictionary to convert labels to class names
        dims (tuple): (width, height) tuple of slide tile dimensions
        mode (str, optional): Either 'save', which saves the visualization to disk or
            'jupyter' which can show the visualization in an IPython console. Defaults
            to 'save'.

    Returns:
        None (output printed or saved to disk based on mode setting)

    """
    path = get_slide_path(slide)
    slide_obj = openslide.OpenSlide(path)
    im = slide_obj.get_thumbnail((dims[1], dims[0])).resize((dims[1], dims[0]))
    dpi = 100
    dims_in = ((dims[1] / dpi) + 1, (dims[0] / dpi) + 1)

    fig = plt.figure(figsize=dims_in, dpi=dpi)
    ax = plt.gca()
    ax.imshow(im, alpha = 0.7)
    arr = ax.imshow(preds_array, interpolation='none', cmap=plt.cm.BrBG, vmin=0, vmax=1, alpha=0.4)
    ax.set_title(f"{slide} Classification Confidences")
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(arr, ticks=[0.0, 0.5, 1.0], cax=cax)
    cbar.set_ticklabels([process_label(label_to_class[0]), '', process_label(label_to_class[1])])
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    if mode == 'jupyter':
        plt.show()
        plt.close([args_0])

    elif mode == 'save':
        viz_dir = f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/visualizations"
        if not os.path.isdir(viz_dir):
            os.makedirs(viz_dir)
        plt.savefig(f"{viz_dir}/{slide}.png")
        plt.close()

def process_label(label):
    """
    Helper function to format class names for nice printing

    Args:
        label (str): the class label to format

    Returns:
        (str): A formatted version of the class label for printing
    """
    return label.replace("_"," ").title()

def get_slide_path(slide):
    """
    Helper function to extract the path to a raw slide image from the slide name

    Args:
        slide (str): the class label to format

    Returns:
        (str): The path to the raw slide file
    """
    if "Scan" in slide:
        path = f"{constants.SLIDE_FILE_DIRECTORY}/" + "/".join(slide.split("_")) + f"/{slide}.qptiff"
    else:
        prefix = re.split("[0-9]+", slide)[0]
        img_num = int(re.findall("[0-9]+", slide)[0])

        path = f"{constants.SLIDE_FILE_DIRECTORY}/{prefix}{img_num:02}/{slide}.svs"
    return path

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

    true_ls_ratio = sa_dict['large_tumor'] / (sa_dict['large_tumor'] + sa_dict['small_tumor'])
    pred_ls_ratio = num_per_class['large_tumor'] / (num_per_class['large_tumor'] + num_per_class['small_tumor'])

    if class_to_label['large_tumor'] == 1:
        mean_conf = np.nanmean(preds_array)
    else:
        mean_conf = 1 - np.nanmean(preds_array)



    print(f"Predicted Large-Small Ratio: {pred_ls_ratio:.2f}; True ratio: {true_ls_ratio:.2f}")
    print(f"Mean Confidence: {mean_conf:.2f}")

    return true_ls_ratio, pred_ls_ratio, mean_conf


def process_predictions(slide):
    """
    Outputs and visualizes the predicted confidences and class labels for a given slide

    Args:
        slide (str): the slide to process

    Returns:
        (defaultdict): Dict containing the true and predicted class surface area ratios
            for each class

    """

    preds_array, coords, dims, df = create_preds_array(slide)
    if constants.KNN_SMOOTH:
        preds_array = knn_smooth(preds_array, coords)

    class_to_label = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/class_to_label")
    label_to_class = {v:k for k,v in class_to_label.items()}

    confusion_matrix = get_confusion_matrix(slide, df)
    print_cm(confusion_matrix, labels = [label_to_class[i] for i in range(max(label_to_class.keys()) + 1)])
    print()

    num_per_class, sa_per_class = estimate_surface_areas(preds_array, label_to_class)

    visualize_predictions(preds_array, slide, label_to_class, dims)

    sa_dict = get_sa_for_slide(slide)
    true, pred, mean_conf = get_metrics(num_per_class, sa_dict, preds_array, class_to_label)

    return true, pred, confusion_matrix, mean_conf

def process_all_predictions():
    """
    Outputs and visualizes the predicted confidences and class labels for all slides
    in the predictions directory

    Args:
        None

    Returns:
        None
    """
    confusion_mat = np.zeros((2, 2), dtype=np.int32)
    df = pd.DataFrame(columns=['slide', 'predicted_ratio', 'true_ratio', 'mean_conf'])

    for slide_file in os.listdir(constants.PREDICTIONS_DIRECTORY):
        if '.csv' not in slide_file or slide_file == "predicted_ratios.csv":
            continue
        slide = slide_file.replace(".csv", "")
        print(f"Results for Slide {slide}")
        true, pred, confuse, mean_conf = process_predictions(slide)

        confusion_mat += confuse
        df = df.append({'slide':slide, 'predicted_ratio':pred, 'true_ratio':true, 'mean_conf':mean_conf}, ignore_index=True)

    df.to_csv(os.path.join(constants.PREDICTIONS_DIRECTORY, "predicted_ratios.csv"))

    class_to_label = load_pickle_from_disk(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/class_to_label")
    label_to_class = {v:k for k,v in class_to_label.items()}

    print("Final Confusion Matrix")
    print()
    print_cm(confusion_mat, labels = [label_to_class[i] for i in range(max(label_to_class.keys()) + 1)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--slide", type=str, required=False, help="Slide for which to output predictions")

    args = parser.parse_args()

    if args.slide is None:
        process_all_predictions()
    else:
        process_predictions(args.slide)
