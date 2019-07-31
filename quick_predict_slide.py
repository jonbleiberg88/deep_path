import constants
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from utils.file_utils import load_pickle_from_disk




def get_slide_metrics(preds_file, label_to_class, method=constants.AGGREGATION_METHOD):
    df = pd.read_csv(preds_file)

    preds = df.prediction.values

    mean_pred = np.mean(preds)

    if method == "mean":
        pred_label = int(round(mean_pred))
    elif method == "mode":
        pred_label = int(round(np.mean(np.round(preds))))
    true_label = df.labels[0]

    return true_label, pred_label, mean_pred

def get_overall_metrics(label_to_class, predict_dir=constants.PREDICTIONS_DIRECTORY, a_vals=np.linspace(0.5,2.0,20)):
    correct_list = []
    wrong_list = []

    true_labels = []
    mean_preds = []
    confusion_mat = np.zeros((2,2), dtype=int)

    preds_files = []
    slide_names = []


    for preds_file in os.listdir(predict_dir):
        if '.csv' not in preds_file:
            continue
        path = os.path.join(predict_dir, preds_file)
        slide_name = preds_file.replace(".csv", "")

        slide_names.append(slide_name)
        preds_files.append(os.path.join(predict_dir, preds_file))

        true_label, pred_label, mean_pred = get_slide_metrics(path, label_to_class)

        true_labels.append(true_label)
        mean_preds.append(mean_pred)

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

    true_labels, mean_preds = np.array(true_labels), np.array(mean_preds)
    roc_auc = roc_auc_score(true_labels, mean_preds)

    acc = np.mean(np.array(correct_list))

    print("_________________________________________________________________")
    print("Misclassified Slides:")
    print(", ".join(wrong_list))

    print("_________________________________________________________________")
    print("Confusion Matrix")
    print_cm(confusion_mat, [label_to_class[0], label_to_class[1]])

    print("_________________________________________________________________")
    print(f"ROC/AUC Score: {roc_auc:.3f}")

    print("_________________________________________________________________")
    print(f"Patient Level Accuracy: {acc*100:.2f}% ({num_correct}/{num_slides})")
    print("_________________________________________________________________")
    print()


    print("Logit Extremizer Correction:")
    corrected_probs, selected_a_vals = logit_extremizer_cv(true_labels, preds_files, a_vals)
    corrected_pred_labels = [int(round(i)) for i in corrected_probs]
    n_correct = 0
    corrected_confusion_mat = np.zeros((2,2), dtype=int)
    metrics = zip(slide_names, mean_preds, corrected_probs, selected_a_vals, true_labels, corrected_pred_labels)

    for slide_name, old_prob, new_prob, a, true_label, pred_label in metrics:
        print(f"Results for Slide {slide_name}:")
        true_class, pred_class = label_to_class[true_label], label_to_class[pred_label]
        if true_class == pred_class:
            n_correct += 1

        corrected_confusion_mat[true_label, pred_label] += 1
        print(f"Predicted Class: {pred_class};  True Class: {true_class}")
        print(f"Uncorrected Confidence: {old_prob:.2f}; Corrected Confidence: {new_prob:.2f}; a: {a}")
        pred_change = np.sign(old_prob - 0.5) != np.sign(new_prob - 0.5)
        print(f"Class Change: {pred_change}")
        print()

    corrected_roc_auc = roc_auc_score(true_labels, corrected_probs)
    corrected_acc = n_correct / num_slides

    print("_________________________________________________________________")
    print("Corrected Confusion Matrix")
    print_cm(corrected_confusion_mat, [label_to_class[0], label_to_class[1]])

    print("_________________________________________________________________")
    print(f"Corrected ROC/AUC Score: {roc_auc:.3f}")

    print("_________________________________________________________________")
    print(f"Corrected Patient Level Accuracy: {acc*100:.2f}% ({n_correct}/{num_slides})")
    print("_________________________________________________________________")



    return acc, num_correct, num_slides, true_labels, mean_preds


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

def logit_extremizer_cv(true_labels, preds_file_paths, a_vals, epsilon=1e-7):
    """
    Use Leave One Out CV to train a logit extremizer model to correct each prediction
    References: https://www.sciencedirect.com/science/article/pii/S0169207013001635
    """
    preds_file_paths = np.array(preds_file_paths)
    true_labels = np.array(true_labels)
    corrected_probs = np.empty(len(true_labels))
    selected_a_vals = np.empty(len(true_labels))

    splitter = LeaveOneOut()
    splits = splitter.split(preds_file_paths)

    for train_idxs, test_idx in splits:
        train_files, train_labels = preds_file_paths[train_idxs], true_labels[train_idxs]
        test_file, test_label = preds_file_paths[test_idx], true_labels[test_idx]
        log_losses = np.zeros(len(a_vals))

        for train_file, train_label in zip(train_files, train_labels):
            corrected_probs = get_corrected_probs_for_slide(train_file, a_vals).astype(np.float32)
            corrected_probs = np.clip(corrected_probs, epsilon, 1-epsilon)
            log_losses += -(train_label * np.log(corrected_probs) + (1 - train_label) * np.log(1 - corrected_probs))

        log_losses /= len(a_vals)
        best_a_val = a_vals[np.argmin(log_losses)]

        selected_a_vals[test_idx] = best_a_val
        corrected_probs[test_idx] = get_corrected_probs_for_slide(test_file, best_a_val)

    return corrected_probs, selected_a_vals

def get_corrected_probs_for_slide(preds_file, a_vals):
    """
    Returns the MLE extremized prediction for a given set of values of "a"
    References: https://www.sciencedirect.com/science/article/pii/S0169207013001635
    """
    df = pd.read_csv(preds_file)
    preds = df.prediction.values

    N = len(preds)
    odds = np.power(np.prod((preds / (1-preds)) ** (1 / N)), a_vals)
    corrected_preds = odds / (1 + odds)

    return corrected_preds



if __name__ == "__main__":
    class_to_label = load_pickle_from_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, 'class_to_label'))
    label_to_class = {v:k for k,v in class_to_label.items()}

    get_overall_metrics(label_to_class, constants.PREDICTIONS_DIRECTORY)
