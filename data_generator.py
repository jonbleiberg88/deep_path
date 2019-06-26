### Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from sklearn.metrics import log_loss
import random
import constants
from collections import defaultdict, Counter
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TrainDataGenerator(tf.keras.utils.Sequence):
    """Loads train images for Keras fit_generator function"""
    def __init__(self, data_dict, batch_size=constants.BATCH_SIZE,
                    batches_per_epoch=constants.BATCHES_PER_EPOCH, out_dim=constants.OUTPUT_IMAGE_DIM,
                    resize=constants.RESIZE_IMAGES, n_channels=constants.N_CHANNELS, n_classes=2,
                    balance_classes=constants.BALANCE_CLASSES, weight_by_size=constants.WEIGHT_BY_SIZE,
                    use_aug=constants.USE_AUGMENTATION):
        'Initialization'

        self.data_dict = data_dict
        self.folders = list(data_dict.keys())

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.epoch_length = batch_size * batches_per_epoch

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_dim = out_dim
        self.resize = resize

        self.balance_classes = balance_classes
        if not self.balance_classes:
            self.merge_classes()

        self.weight_by_size = weight_by_size
        if self.weight_by_size:
            self.get_weights()

        self.use_aug = use_aug
        if self.use_aug:
            self.aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                            height_shift_range=0.1,horizontal_flip=True,
                                            vertical_flip=True, data_format='channels_last',
                                            fill_mode='reflect',
                                            zoom_range=[0.9, 1.25])

        self.paths_for_epoch = []
        self.labels_for_epoch = []
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_ids = range(index * self.batch_size,(index+ 1) * self.batch_size)

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def on_epoch_end(self):
        self.paths_for_epoch = []
        self.labels_for_epoch = []

        if self.balance_classes:
            for class_name, class_dict in self.data_dict.items():
                if self.weight_by_size:
                    epoch_folders = random.choices(list(class_dict.keys()),
                        k = int(self.epoch_length / self.n_classes), weights=self.weights[class_name])
                else:
                    epoch_folders = random.choices(list(class_dict.keys()),
                        k = int(self.epoch_length / self.n_classes))
                for dir in epoch_folders:
                    folder_data = class_dict[dir]
                    path, label = random.choice(folder_data)
                    self.paths_for_epoch.append(path)
                    self.labels_for_epoch.append(label)
        else:
            if self.weight_by_size:
                epoch_folders = random.choices(list(self.data_dict.keys()),
                    k = self.epoch_length, weights=self.weights)
            else:
                epoch_folders = random.choices(list(self.data_dict.keys()),
                    k = self.epoch_length)
            for dir in epoch_folders:
                folder_data = self.data_dict[dir]
                path, label = random.choice(folder_data)
                self.paths_for_epoch.append(path)
                self.labels_for_epoch.append(label)

        self.paths_for_epoch = np.array(self.paths_for_epoch)
        self.labels_for_epoch = np.array(self.labels_for_epoch, dtype=int)

        shuffle_idxs = np.random.permutation(np.arange(len(self.paths_for_epoch)))

        self.paths_for_epoch = self.paths_for_epoch[shuffle_idxs]
        self.labels_for_epoch = self.labels_for_epoch[shuffle_idxs]


    def __data_generation(self, batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *out_dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.out_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, idx in enumerate(batch_ids):
            # Store image
            X[i,] = self.get_img(self.paths_for_epoch[idx])

            # Store class
            y[i] = self.labels_for_epoch[idx]

        if self.n_classes > 2:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y

    def get_img(self, path):
        # im = Image.open(path)
        # if self.resize:
        #     im = im.resize(self.out_dim)
        # return (np.array(im) / 127.5).astype(np.float32) -1.

        im = tf.keras.preprocessing.image.load_img(path, target_size=self.out_dim)
        im = tf.keras.preprocessing.image.img_to_array(im)
        if self.use_aug:
            im = self.aug.random_transform(im)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)

        return np.squeeze(im)

    def merge_classes(self):
        new_dict = defaultdict(list)

        for class_dict in self.data_dict.values():
            for slide, data_list in class_dict.items():
                new_dict[slide].append(data_list)

        self.data_dict = new_dict

    def get_weights(self):
    # Requires Python 3.6 or above to maintain dict ordering....
        if self.balance_classes:
            self.weights = defaultdict(list)
            for img_class, class_dict in self.data_dict.items():
                for slide_data in class_dict.values():
                    self.weights[img_class].append(len(slide_data))

        else:
            self.weights = []
            for slide_data in self.data_dict.values():
                self.weights.append(len(slide_data))





class ValDataGenerator(tf.keras.utils.Sequence):
    """Loads validation or test images for Keras fit_generator function"""
    def __init__(self, data_dict, batch_size=constants.BATCH_SIZE, resize=constants.RESIZE_IMAGES,
                    out_dim=constants.OUTPUT_IMAGE_DIM, n_channels=constants.N_CHANNELS,
                    n_classes=2, shuffle=True):
        'Initialization'
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.out_dim = out_dim

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize = resize

        self.use_aug = constants.USE_TTA
        if self.use_aug:
            self.aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                             height_shift_range=0.1,horizontal_flip=True,
                                             vertical_flip=True, data_format='channels_last',
                                             fill_mode='reflect',
                                             zoom_range=[0.9, 1.25])

        self.extract_paths_and_labels()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_paths = self.paths[indexes]
        batch_labels = self.labels[indexes]

        # Generate data
        X, y = self.__data_generation(batch_paths, batch_labels)
        # print("get item")

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_labels):
        'Generates data containing batch_size samples' # X : (n_samples, *out_dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.out_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, data in enumerate(zip(batch_paths, batch_labels)):
            path, label = data
            # Store sample
            X[i,] = self.get_img(path)

            # Store class
            y[i] = label

        if self.n_classes > 2:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def get_img(self, path):
        # im = Image.open(path)
        # if self.resize:
        #     im = im.resize(self.out_dim)
        # return (np.array(im) / 127.5).astype(np.float32) - 1.

        im = tf.keras.preprocessing.image.load_img(path, target_size=self.out_dim)
        im = tf.keras.preprocessing.image.img_to_array(im)
        if self.use_aug:
            im = self.aug.random_transform(im)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        return np.squeeze(im)

    def extract_paths_and_labels(self):
        self.paths = []
        self.labels = []
        for class_dict in self.data_dict.values():
            for slide_data_list in class_dict.values():
                for path, label in slide_data_list:
                    self.paths.append(path)
                    self.labels.append(label)

        self.paths = np.array(self.paths)
        self.labels = np.array(self.labels, dtype =int)


class TestDataGenerator(tf.keras.utils.Sequence):
    """ Generates data batches for test set prediction, optionally with test time augmentation"""
    def __init__(self, data_dict, use_tta=constants.USE_TTA, aug_times=constants.TTA_AUG_TIMES,
                    batch_size=constants.BATCH_SIZE, resize=constants.RESIZE_IMAGES,
                    out_dim=constants.OUTPUT_IMAGE_DIM, n_channels=constants.N_CHANNELS,
                    n_classes=2):
        'Initialization'
        self.data_dict = data_dict

        self.use_tta = use_tta
        if self.use_tta:
            self.aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                            height_shift_range=0.1,horizontal_flip=True,
                                            vertical_flip=True, data_format='channels_last',
                                            fill_mode='reflect',
                                            zoom_range=[0.9, 1.25])
            self.aug_times = aug_times

        self.batch_size = batch_size
        self.out_dim = out_dim

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize = resize

        self.extract_paths_and_labels()
        self.on_epoch_end()
        print(self.__len__())
        print(f"Paths: {len(self.paths)}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index + 1) * self.batch_size > len(self.indexes):
            indexes = self.indexes[index*self.batch_size:-1]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_paths = self.paths[indexes]
        batch_labels = self.labels[indexes]

        # Generate data
        X, y = self.__data_generation(batch_paths, batch_labels)
        if index > 180:
            print(batch_paths)
            print(X.shape)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))

    def __data_generation(self, batch_paths, batch_labels):
        'Generates data containing batch_size samples' # X : (n_samples, *out_dim, n_channels)
        # Initialization
        size = len(batch_paths)
        X = np.empty((size, *self.out_dim, self.n_channels))
        y = np.empty((size), dtype=int)

        # Generate data
        for i, data in enumerate(zip(batch_paths, batch_labels)):
            path, label = data
            # Store sample
            X[i,] = self.get_img(path)

            # Store class
            y[i] = label

        if self.n_classes > 2:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def get_img(self, path):
        # im = Image.open(path)
        # if self.resize:
        #     im = im.resize(self.out_dim)
        # return (np.array(im) / 127.5).astype(np.float32) - 1.

        im = tf.keras.preprocessing.image.load_img(path, target_size=self.out_dim)
        im = tf.keras.preprocessing.image.img_to_array(im)
        if self.use_tta:
            im = self.aug.random_transform(im)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        return np.squeeze(im)

    def extract_paths_and_labels(self):
        self.paths = []
        self.labels = []
        if self.use_tta:
            self.unique_paths, self.unique_labels = [], []

        for class_dict in self.data_dict.values():
            for slide_data_list in class_dict.values():
                for path, label in slide_data_list:
                    if self.use_tta:
                        for _ in range(self.aug_times):
                            self.paths.append(path)
                            self.labels.append(label)
                        self.unique_paths.append(path)
                        self.unique_labels.append(label)

                    else:
                        self.paths.append(path)
                        self.labels.append(label)

        if len(self.paths) % self.batch_size == 2:
            self.paths.append(self.paths[-1])
            self.labels.append(self.labels[-1])
            print('appending')

        self.paths = np.array(self.paths)
        self.labels = np.array(self.labels, dtype=int)



        if self.use_tta:
            self.unique_paths = np.array(self.unique_paths)
            self.unique_labels = np.array(self.unique_labels)

    def extract_TTA_preds(self, preds, return_dict=False):
        predict_lists = {p:[] for p in self.unique_paths}
        predictions = {p:0 for p in self.unique_paths}

        for pred, path in zip(preds, self.paths):
             predict_lists[path].append(pred)

        for path, pred_list in predict_lists.items():
            pred = np.mean(pred_list)
            predictions[path] = pred

        if return_dict:
            return predictions
        else:
            items = list(predictions.items())
            paths = np.array([path for path, _ in items])
            preds = np.array([pred for _,pred in items])

            return paths, preds

    def eval(self, preds):
        if self.use_tta:
            loss = log_loss(self.unique_labels, preds,labels=[0,1], eps=1e-8)
            pred_class = np.rint(preds)
            accuracy = np.mean(pred_class == self.unique_labels)


            return loss, accuracy

        else:
            loss = log_loss(self.labels, preds, labels=[0,1], eps=1e-8)
            pred_class = np.rint(preds)
            accuracy = np.mean(pred_class == self.labels)

            return loss, accuracy
