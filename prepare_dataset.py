import os
import numpy as np
import pandas as pd
import constants

from collections import defaultdict
from sklearn.model_selection import KFold

def get_dataset_for_fold(data_dir, folds_list, fold, class_to_label, slide_to_label):
    """
    Given the root directory holding the dataset, and the train test split,
    gets paths and creates labels for all of the images in the train and test sets

    Args:
        data_dir (String): Path to top-level directory of dataset
        folds_list (dict of dicts): Output of split_train_test
        fold (int): Fold for which to extract data
    Returns:
        dict containing the paths, labels for train images in the form
            train_dict[SLIDE_FOLDER] = [(PATH, LABEL), ...]
        dict containing the paths, labels for test images in the form
            test_dict[SLIDE_FOLDER] = [(PATH, LABEL), ...]
        dict to convert between class names and integer labels
    """
    train_slides = [s for s, _ in folds_list[fold]['train']]
    test_slides = [s for s, _ in folds_list[fold]['test']]

    train_dict = defaultdict(lambda: defaultdict(list))
    test_dict = defaultdict(lambda: defaultdict(list))

    for orig_class in os.listdir(data_dir):
        orig_class_path = os.path.join(data_dir, orig_class)
        slide_folders = os.listdir(orig_class_path)

        for slide in slide_folders:
            if slide in train_slides:
                slide_path = os.path.join(orig_class_path, slide)
                for img in os.listdir(slide_path):
                    if img.endswith('.jpg'):
                        path = os.path.join(slide_path, img)
                        label = slide_to_label[slide]
                        train_dict[label][slide].append((path, label))
            elif slide in test_slides:
                slide_path = os.path.join(orig_class_path, slide)
                for img in os.listdir(slide_path):
                    if img.endswith('.jpg'):
                        path = os.path.join(slide_path, img)
                        label = slide_to_label[slide]
                        test_dict[label][slide].append((path, label))
            else:
                print(f"{slide} not assigned to train or test...")


        for class_name, class_dict in train_dict.items():
            for slide in list(class_dict.keys()):
                if len(class_dict[slide]) == 0:
                    del train_dict[class_name][slide]

        for class_name, class_dict in test_dict.items():
            for slide in list(class_dict.keys()):
                if len(class_dict[slide]) == 0:
                    del test_dict[class_name][slide]

        if constants.STRATIFY:
            for class_name, class_dict in train_dict.items():
                print(f"{class_name}: {len(list(class_dict.keys()))}")

    return train_dict, test_dict


def get_full_dataset(data_dir, slide_names, class_to_label, slide_to_label):
    """
    Given the root directory holding the dataset and a list of slides,
    gets paths and creates labels for all of the images

    Args:
        data_dir (String): Path to top-level directory of dataset
        folds_list (dict of dicts): Output of split_train_test
        fold (int): Fold for which to extract data
    Returns:
        dict containing the paths, labels for train images in the form
            train_dict[CLASS_LABEL][SLIDE_FOLDER] = [(PATH, LABEL), ...]

        dict to convert between class names and integer labels
    """

    train_dict = defaultdict(lambda: defaultdict(list))

    for orig_class in os.listdir(data_dir):
        orig_class_path = os.path.join(data_dir, orig_class)
        slide_folders = os.listdir(orig_class_path)

        for slide in slide_folders:
            if slide in slide_names:
                slide_path = os.path.join(orig_class_path, slide)
                for img in os.listdir(slide_path):
                    if img.endswith('.jpg'):
                        path = os.path.join(slide_path, img)
                        label = slide_to_label[slide]
                        train_dict[label][slide].append((path, label))


        for class_name, class_dict in train_dict.items():
            for slide in list(class_dict.keys()):
                if len(class_dict[slide]) == 0:
                    del train_dict[class_name][slide]

    return train_dict



def split_train_test(data_dir, num_folds, label_file = constants.LABEL_FILE,
                        verbose=True, stratified=constants.STRATIFY):
    """
    Given the root directory holding the dataset and a number of folds, splits the dataset
    into train and test set by patient

    Args:
        data_dir (String): Path to top-level directory of dataset
        num_folds (int): Number of folds for cross validation
        verbose (boolean): Whether to print statistics on the split
    Returns:
        folds_list (dict of dicts): Split of slide folder into train and test set for each
            fold, in the format folds_list[fold_number]['train' or 'test'] = [SLIDE NAMES,...]
    """

    labels_df = pd.read_csv(label_file)
    label_set = set(labels_df.iloc[:,1])
    class_to_label = {c:idx for idx, c in enumerate(label_set)}
    slide_to_class = dict(zip(labels_df.iloc[:,0], labels_df.iloc[:,1]))
    slide_to_label = {slide: class_to_label[c] for slide, c in slide_to_class.items()}


    if stratified:
        image_counts = get_class_counts_for_images(data_dir)
        class_lists = create_class_lists(image_counts)

        folds_list = [{'train':[], 'test': []} for _ in range(num_folds)]

        for class_name, class_list in class_lists.keys():
            img_list = [(img, class_name) for img in class_list]
            slide_to_label = {img:class_to_label[c] for img, c in img_list}

            kf = KFold(n_splits=num_folds, shuffle=True)
            split = list(kf.split(img_list))
            for idx, split in enumerate(split):
                folds_list[idx]['train'] += list(np.array(img_list)[split[0]])
                folds_list[idx]['test'] += list(np.array(img_list)[split[1]])

        if verbose:
            print_class_counts(folds_list, image_counts, slide_to_class)
    else:
        img_list = []
        folds_list = [{'train':[], 'test': []} for _ in range(num_folds)]

        for class_dir in os.listdir(data_dir):
            full_path = os.path.join(data_dir, class_dir)
            img_list += [slide for slide in os.listdir(full_path) if slide.split("_")[0] in slide_to_class.keys()]

        img_list = [(img, slide_to_class[img.split("_")[0]]) for img in list(set(img_list))]
        slide_to_label = {img:class_to_label[c] for img, c in img_list}

        kf = KFold(n_splits=num_folds, shuffle=True)
        split = list(kf.split(img_list))
        for idx, split in enumerate(split):
            folds_list[idx]['train'] += list(np.array(img_list)[split[0]])
            folds_list[idx]['test'] += list(np.array(img_list)[split[1]])

        if verbose:
            image_counts = get_counts_for_images(data_dir)
            print_class_counts(folds_list, image_counts, slide_to_class)

    return folds_list, class_to_label, slide_to_label

def get_counts_for_images(data_dir):
    """
    Given the root directory holding the dataset, calculates the number of images
    for each slide

    Args:
        data_dir (String): Path to top-level directory of dataset
    Returns:
        dict of dicts containing the number of images per slide, in the
        format: image_class_counts[SLIDE] = # of images
    """
    orig_class_dirs = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    image_counts = defaultdict(int)

    for dir in orig_class_dirs:
        full_path = os.path.join(data_dir, dir)
        image_dirs = [(img, os.path.join(full_path, img)) for img in os.listdir(full_path)]
        for img, path in image_dirs:
            image_counts[img] += len(os.listdir(path))

    return image_counts

def create_class_lists(image_counts, slide_to_class):
    """
    Given counts of images per slide per class (see get_counts_for_images),
    assigns each patient to a class for the purposes of stratified splitting

    Args:
        image_class_counts (dict of dicts): Output of get_class_counts_for_images
    Returns:
        dict of form image_lists[CLASS] = [SLIDES_FOR_CLASS,...]
    """
    image_lists = defaultdict(list)

    for img, count in image_counts.items():
        slide_name = img.split("_")[0]
        if slide_name in slide_to_class.keys():
            image_lists[slide_to_class[slide_name]].append(img)

    return image_lists

def print_class_counts(folds_list, image_counts, slide_to_class):
    """
    Given the folds_list (see split_train_test), and the number of images of each class per slide
    (see get_class_counts_for_images), nicely prints train and test statistics to
    the console

    Args:
        folds_list (dict of dicts): Output of split_train_test
        image_class_counts (dict of dicts): Output of get_class_counts_for_images
        num_classes (int): Number of classes in the dataset
    Returns:
        None (prints output to console)
    """
    class_counts = [{'train':defaultdict(int), 'test':defaultdict(int)} for _ in folds_list]

    for idx, fold in enumerate(folds_list):
        train_imgs = fold['train']
        test_imgs = fold['test']
        for img, class_name in train_imgs:
            count = image_counts[img]
            class_counts[idx]['train'][class_name] += count

        for img, class_name in test_imgs:
            count = image_counts[img]
            class_counts[idx]['test'][class_name] += count

    for idx, fold in enumerate(class_counts):
        print(f"Fold {idx}")
        for set, counts in fold.items():
            print(f"{set.title()}:")
            for name, num in counts.items():
                print(f"{name.title()}: {num}")

        print("_______________________________________")
