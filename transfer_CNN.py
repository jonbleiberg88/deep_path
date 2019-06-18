import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import multi_gpu_model

class TransferCNN:
    def __init__(self, input_shape=(256,256,3), base_model=ResNet50,layer_sizes=[512],
        n_classes=2, use_bn=True, use_dropout=False,
        optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9,nesterov=True), metrics=['accuracy']):
        self.input_shape = input_shape
        self.base_model = base_model(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling='max')
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = None


    def init_model(self):
        layer_list = [self.base_model]
        for units in self.layer_sizes:
            layer_list.append(Dense(units, activation='relu'))
            if self.use_bn:
                layer_list.append(BatchNormalization())
            if self.use_dropout:
                layer_list.append(Dropout())

        # if self.n_classes == 2:
        #     layer_list.append(Dense(1, activation='sigmoid'))
        # elif self.n_classes > 2:
        layer_list.append(Dense(self.n_classes, activation='softmax'))

        self.set_trainable(False)
        model = Sequential(layer_list)
        self.model = model


        return model

    def compile_model(self):
        if self.model is None:
            self.init_model()
        #self.model = multi_gpu_model(self.model, gpus=4)
        #elf.n_classes > 2:
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=self.metrics)

        print(self.model.summary())
        return self.model

    def set_trainable(self, trainable):
        for layer in self.base_model.layers:
            layer.trainable = trainable
