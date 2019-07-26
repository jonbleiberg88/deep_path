import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
# from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K

import constants
# SGD(lr=0.2, decay=1e-6, momentum=0.9,nesterov=True)

class TransferCNN:
    def __init__(self, input_shape=constants.INPUT_SHAPE, base_model=MobileNet,layer_sizes=constants.LAYER_SIZES,
        n_classes=constants.NUM_CLASSES, use_bn=constants.USE_BATCH_NORM, use_dropout=constants.USE_DROPOUT,
        optimizer='adam', metrics=constants.METRICS):

        if constants.CAP_MEMORY_USAGE:
            # https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
            # TensorFlow wizardry
            config = tf.ConfigProto()

            # Don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True

            # Only allow a total of half the GPU memory to be allocated
            config.gpu_options.per_process_gpu_memory_fraction = 0.5

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

            # Create a session with the above options specified.
            K.set_session(tf.Session(config=config))

        self.input_shape = input_shape
        self.base_model = base_model(weights='imagenet', include_top=False, input_shape=input_shape,pooling=constants.OUTPUT_POOLING)
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None


    def init_model(self):
        layer_list = [self.base_model, Flatten(), BatchNormalization()]
        for units in self.layer_sizes:
            layer_list.append(Dense(units, activation='relu'))
            if self.use_bn:
                layer_list.append(BatchNormalization())
            if self.use_dropout:
                layer_list.append(Dropout(rate=constants.DROPOUT_RATE))

        if self.n_classes == 2:
            layer_list.append(Dense(1, activation='sigmoid'))
        if self.n_classes > 2:
            layer_list.append(Dense(self.n_classes, activation='softmax'))

        if constants.FREEZE:
            self.set_trainable(False)
        model = Sequential(layer_list)
        self.model = model


        return model

    def compile_model(self):
        if self.model is None:
            self.init_model()
        self.base_model = self.model

        if constants.GPUS > 1:
            self.model = multi_gpu_model(self.base_model, gpus=constants.GPUS, cpu_merge=False)
        if self.n_classes == 2:
            self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=self.metrics)
        if self.n_classes > 2:
            self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=self.metrics)

        return self.model, self.base_model

    def set_trainable(self, trainable):
        for layer in self.base_model.layers:
            layer.trainable = trainable
