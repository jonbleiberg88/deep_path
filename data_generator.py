### Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
#import constants

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, out_dim=(256,256), n_channels=3,
                 n_classes=2, shuffle=True, resize=False):
        'Initialization'
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize = resize
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *out_dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.out_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.get_img(ID)

            # Store class
            y[i] = self.labels[i]

        if self.n_classes != 2:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def get_img(self, path):
        im = Image.open(path)
        if self.resize:
            im = im.resize(self.out_dim)
        return np.array(im) / 256
