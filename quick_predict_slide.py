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

    for preds_file in os.listdir(predict_dir):
        if '.csv' not in preds_file:
            continue
        path = os.path.join(predict_dir, preds_file)
        slide_name = preds_file.replace(".csv", "")

        true_label, pred_label, mean_pred = get_slide_metrics(path, label_to_class)

        true_class, pred_class = label_to_class[true_label], label_to_class[pred_label]

        correct = (true_label == pred_label)
        correct.append(true_label == pred_label)

        print(f"Results for Slide {slide_name}:")

        print(f"Predicted Class: {pred_class};  True Class: {true_class}; Mean Confidence: {mean_pred:.2f}")
        print()




    num_slides = len(correct)
    num_correct = len([i for i in correct if i])

    acc = np.mean(np.array(correct))

    print("_________________________________________________________________")
    print(f"Patient Level Accuracy: {acc:.2f} ({num_correct}/{num_slides})")
    print("_________________________________________________________________")
    return acc, num_correct, num_slides

if __name__ == "__main__":
    class_to_label = load_pickle_from_disk(os.path.join(constants.VISUALIZATION_HELPER_FILE_FOLDER, 'class_to_label'))
    label_to_class = {v:k for k,v in class_to_label.items()}

    get_overall_metrics(label_to_class, constants.PREDICTIONS_DIRECTORY)
