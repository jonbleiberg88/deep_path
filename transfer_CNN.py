import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import multi_gpu_model

import constants
# SGD(lr=0.2, decay=1e-6, momentum=0.9,nesterov=True)

class TransferCNN:
    def __init__(self, input_shape=constants.INPUT_SHAPE, base_model=ResNet50,layer_sizes=constants.LAYER_SIZES,
        n_classes=2, use_bn=constants.USE_BATCH_NORM, use_dropout=constants.USE_DROPOUT,
        optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True), metrics=constants.METRICS):
        self.input_shape = input_shape
        self.base_model = base_model(weights='imagenet', include_top=False, pooling=constants.OUTPUT_POOLING)
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
                layer_list.append(Dropout())

        if self.n_classes == 2:
            layer_list.append(Dense(1, activation='sigmoid'))
        if self.n_classes > 2:
            layer_list.append(Dense(self.n_classes, activation='softmax'))

        # self.set_trainable(False)
        model = Sequential(layer_list)
        self.model = model


        return model

    def compile_model(self):
        if self.model is None:
            self.init_model()
        if constants.GPUS > 1:
            self.model = multi_gpu_model(self.model, gpus=constants.GPUS)
        if self.n_classes == 2:
            self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=self.metrics)
        if self.n_classes > 2:
            self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=self.metrics)

        print(self.model.summary())
        return self.model

    def set_trainable(self, trainable):
        for layer in self.base_model.layers:
            layer.trainable = trainable
