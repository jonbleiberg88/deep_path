### Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import random
import constants


class TrainDataGenerator(tf.keras.utils.Sequence):
    """Loads train images for Keras fit_generator function"""
    def __init__(self, data_dict, batch_size=64, batches_per_epoch=400, out_dim=(256,256), n_channels=3,
                 n_classes=2, shuffle=True, resize=False):
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
        self.batch_num += 1

        return X, y

    def on_epoch_end(self):
        self.paths_for_epoch = []
        self.labels_for_epoch = []
        self.batch_num = 0

        epoch_folders = random.choices(self.folders, k = self.epoch_length)
        for dir in epoch_folders:
            folder_data = self.data_dict[dir]
            path, label = random.choice(folder_data)
            self.paths_for_epoch.append(path)
            self.labels_for_epoch.append(label)
        print(len(self.paths_for_epoch))
        print(len(self.labels_for_epoch))

        self.paths_for_epoch = np.array(self.paths_for_epoch)
        self.labels_for_epoch = np.array(self.labels_for_epoch, dtype=np.float32)

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

        if self.n_classes != 2:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y

    def get_img(self, path):
        # im = tf.keras.preprocessing.image.load_img(path)
        # return tf.keras.preprocessing.image.img_to_array(im)
        # tf.keras.applications.resnet50.preprocess_input()
        im = Image.open(path)
        if self.resize:
            im = im.resize(self.out_dim)
        return (np.array(im) / 255.).astype(np.float32)

class ValDataGenerator(tf.keras.utils.Sequence):
    """Loads validation or test images for Keras fit_generator function"""
    def __init__(self, data_dict, batch_size=64, out_dim=(256,256), n_channels=3,
                 n_classes=2, shuffle=True, resize=False):
        'Initialization'
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.out_dim = out_dim

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize = resize

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

        if self.n_classes != 2:
            y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def get_img(self, path):
        im = Image.open(path)
        if self.resize:
            im = im.resize(self.out_dim)
        return (np.array(im) / 255.).astype(np.float32)

    def extract_paths_and_labels(self):
        self.paths = []
        self.labels = []

        for slide_data_list in self.data_dict.values():
            for path, label in slide_data_list:
                self.paths.append(path)
                self.labels.append(label)
        self.paths = np.array(self.paths)
        self.labels = np.array(self.labels, dtype =int)
