import constants
import os
import numpy as np
import pandas as pd

from utils.file_utils import load_pickle_from_disk



def get_slide_metrics(preds_file, label_to_class, method=constants.AGGREGATION_METHOD):
    df = pd.read_csv(preds_file)

    preds = df.prediction.values

    mean_pred = np.mean(preds)

    if method == "mean":
        pred_label = int(mean_pred)
    elif method == "mode":
        pred_label = int(np.mean(np.round(preds)))
    true_label = df.labels[0]


    return true_label, pred_label, mean_pred

def get_overall_metrics(label_to_class, predict_dir=constants.PREDICTIONS_DIRECTORY):
    correct_list = []
    wrong_list = []
    confusion_mat = np.zeros((2,2), dtype=int)

    for preds_file in os.listdir(predict_dir):
        if '.csv' not in preds_file:
            continue
        path = os.path.join(predict_dir, preds_file)
        slide_name = preds_file.replace(".csv", "")

        true_label, pred_label, mean_pred = get_slide_metrics(path, label_to_class)

        true_class, pred_class = label_to_class[true_label], label_to_class[pred_label]

        confusion_mat[true_label, pred_label] += 1

        correct = (true_label == pred_label)
        correct_list.append(true_label == pred_label)
        if not correct:
            wrong_list.append(slide_name)

        print(f"Results for Slide {slide_name}:")

        print(f"Predicted Class: {pred_class};  True Class: {true_class}; Mean Confidence: {mean_pred:.2f}")
        print()




    num_slides = len(correct_list)
    num_correct = len([i for i in correct_list if i])

    acc = np.mean(np.array(correct_list))

    print("_________________________________________________________________")
    print("Misclassified Slides:")
    print(", ".join(wrong_list))

    print("_________________________________________________________________")
    print("Confusion Matrix")
    print_cm(confusion_mat, [label_to_class[0], label_to_class[1]])

    print("_________________________________________________________________")
    print(f"Patient Level Accuracy: {acc*100:.2f}% ({num_correct}/{num_slides})")
    print("_________________________________________________________________")
    return acc, num_correct, num_slides


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


if __name__ == "__main__":
    class_to_label = load_pickle_from_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, 'class_to_label'))
    label_to_class = {v:k for k,v in class_to_label.items()}

    get_overall_metrics(label_to_class, constants.PREDICTIONS_DIRECTORY)
