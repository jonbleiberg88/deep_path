import os
import numpy as np
import constants

from collections import defaultdict
from sklearn.model_selection import KFold

def get_dataset_for_fold(data_dir, folds_list, fold):
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
    train_slides = folds_list[fold]['train']
    test_slides = folds_list[fold]['test']

    train_dict = defaultdict(lambda: defaultdict(list))
    test_dict = defaultdict(lambda: defaultdict(list))

    classes =  os.listdir(data_dir)
    class_to_label = {c:i for i,c in enumerate(classes)}

    for img_class in classes:
        class_path = os.path.join(data_dir, img_class)
        class_idx = class_to_label[img_class]
        slide_folders = os.listdir(class_path)

        for slide in slide_folders:
            if slide in train_slides:
                slide_path = os.path.join(class_path, slide)
                for img in os.listdir(os.path.join(class_path, slide)):
                    if img.endswith('.jpg'):
                        path = os.path.join(slide_path, img)
                        label = class_idx
                        train_dict[img_class][slide].append((path, label))
            elif slide in test_slides:
                slide_path = os.path.join(class_path, slide)
                for img in os.listdir(os.path.join(class_path, slide)):
                    if img.endswith('.jpg'):
                        path = os.path.join(slide_path, img)
                        label = class_idx
                        test_dict[img_class][slide].append((path, label))
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

        for class_name, class_dict in train_dict.items():
            print(f"{class_name}: {len(list(class_dict.keys()))}")

    return train_dict, test_dict, class_to_label

# def get_dataset_for_fold(data_dir, folds_list, fold):
#     """
#     Given the root directory holding the dataset, and the train test split,
#     gets paths and creates labels for all of the images in the train and test sets
#
#     Args:
#         data_dir (String): Path to top-level directory of dataset
#         folds_list (dict of dicts): Output of split_train_test
#         fold (int): Fold for which to extract data
#     Returns:
#         dict of dicts containing the paths to images in the form
#             data['train' or 'test'] = np.array(IMG_PATHS,...)
#         dict of dicts containing the integer labels for images in the form
#             labels['train' or 'test'] = np.array(IMG_LABELS,...)
#         dict to convert between class names and integer labels
#     """
#     x_train, train_labels = [], []
#     x_test, test_labels = [], []
#
#     classes =  os.listdir(data_dir)
#     class_to_label = {c:i for i,c in enumerate(classes)}
#
#     for img_class in classes:
#         class_path = os.path.join(data_dir, img_class)
#         class_idx = class_to_label[img_class]
#         slide_folders = os.listdir(class_path)
#
#         for slide in slide_folders:
#             if slide in folds_list[0]['train']:
#                 slide_path = os.path.join(class_path, slide)
#                 for img in os.listdir(os.path.join(class_path, slide)):
#                     if img.endswith('.jpg'):
#                         x_train.append(os.path.join(slide_path, img))
#                         train_labels.append(class_idx)
#             elif slide in folds_list[0]['test']:
#                 slide_path = os.path.join(class_path, slide)
#                 for img in os.listdir(os.path.join(class_path, slide)):
#                     if img.endswith('.jpg'):
#                         x_test.append(os.path.join(slide_path, img))
#                         test_labels.append(class_idx)
#             else:
#                 print(f"{slide} not assigned to train or test...")
#
#     x_train = np.array(x_train)
#     x_test = np.array(x_test)
#
#     train_labels = np.array(train_labels, dtype=np.int8)
#     test_labels = np.array(test_labels, dtype=np.int8)
#
#     rand_perm_train = np.random.permutation(x_train.shape[0])
#     rand_perm_test = np.random.permutation(x_test.shape[0])
#
#     x_train, train_labels = x_train[rand_perm_train], train_labels[rand_perm_train]
#
#     x_test, test_labels = x_test[rand_perm_test], test_labels[rand_perm_test]
#
#     data = {'train':x_train, 'test':x_test}
#     labels = {'train': train_labels, 'test':test_labels}
#
#     return data, labels, class_to_label


def split_train_test(data_dir, num_folds, verbose=True, stratified=constants.STRATIFY):
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
    if stratified:
        image_class_counts = get_class_counts_for_images(data_dir)
        class_assignments = assign_folders_to_class(image_class_counts)
        num_classes = len(class_assignments.keys())

        folds_list = [{'train':[], 'test': []} for _ in range(num_folds)]


        for class_name in class_assignments.keys():
            img_list = class_assignments[class_name]
            kf = KFold(n_splits=num_folds, shuffle=True)
            split = list(kf.split(img_list))
            for idx, split in enumerate(split):
                folds_list[idx]['train'] += list(np.array(img_list)[split[0]])
                folds_list[idx]['test'] += list(np.array(img_list)[split[1]])

        if verbose:
            print_class_counts(folds_list, image_class_counts, num_classes)
    else:
        img_list = []
        folds_list = [{'train':[], 'test': []} for _ in range(num_folds)]
        for class_dir in os.listdir(data_dir):
            full_path = os.path.join(data_dir, class_dir)
            img_list += os.listdir(full_path)

        img_list = list(set(img_list))
        kf = KFold(n_splits=num_folds, shuffle=True)
        split = list(kf.split(img_list))
        for idx, split in enumerate(split):
            folds_list[idx]['train'] += list(np.array(img_list)[split[0]])
            folds_list[idx]['test'] += list(np.array(img_list)[split[1]])


    return folds_list

def get_class_counts_for_images(data_dir):
    """
    Given the root directory holding the dataset, calculates the number of images
    per class for each slide

    Args:
        data_dir (String): Path to top-level directory of dataset
    Returns:
        dict of dicts containing the number of images per class per slide, in the
        format: image_class_counts[SLIDE][CLASS] = # of images
    """
    class_dirs = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    image_class_counts = defaultdict(lambda: {dir:0 for dir in class_dirs})

    for dir in class_dirs:
        full_path = os.path.join(data_dir, dir)
        image_dirs = [(img ,os.path.join(full_path, img)) for img in os.listdir(full_path)]
        for img, path in image_dirs:
            image_class_counts[img][dir] += len(os.listdir(path))

    return image_class_counts

def assign_folders_to_class(image_class_counts):
    """
    Given counts of images per slide per class (see get_class_counts_for_images),
    assigns each patient to a class for the purposes of stratified splitting

    Args:
        image_class_counts (dict of dicts): Output of get_class_counts_for_images
    Returns:
        dict of form image_lists[CLASS] = [SLIDES_FOR_CLASS,...]
    """
    image_lists = defaultdict(list)
    class_counts = defaultdict(int)
    for img in image_class_counts.keys():
        count_dict = image_class_counts[img]
        max_class = max(count_dict.items(), key=lambda x: x[1])[0]
        image_lists[max_class].append(img)

    return image_lists

def print_class_counts(folds_list, image_class_counts, num_classes):
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
        for img in train_imgs:
            counts = image_class_counts[img]
            for key in counts.keys():
                class_counts[idx]['train'][key] += counts[key]

        for img in test_imgs:
            counts = image_class_counts[img]
            for key in counts.keys():
                class_counts[idx]['test'][key] += counts[key]

    for idx, fold in enumerate(class_counts):
        print(f"Fold {idx}")
        for set, counts in fold.items():
            print(f"{set.title()}:")
            for name, num in counts.items():
                print(f"{name.title()}: {num}")

        print("_______________________________________")
