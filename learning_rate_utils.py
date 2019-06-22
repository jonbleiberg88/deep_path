import constants

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
from math import ceil

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=constants.BATCHES_PER_EPOCH,
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 epochs= constants.EPOCHS,
                 steps_per_epoch=constants.BATCHES_PER_EPOCH,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay


        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.calculate_lrs()

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the max value at the start of training.'''
        self.idx = 1
        K.set_value(self.model.optimizer.lr, self.lrs[0])

    def on_batch_end(self, batch, logs={}):
        '''Update the learning rate.'''
        K.set_value(self.model.optimizer.lr, self.lrs[self.idx])
        self.idx += 1

    def calculate_lrs(self):
        if self.mult_factor == 1:
            n_cycles = ceil(self.epochs / self.cycle_length)

            fracs = np.tile(np.linspace(0, 1, num=self.steps_per_epoch * self.epochs), n_cycles)
            max_lrs = np.repeat([self.lr_decay ** i for i in range(n_cycles)],
                                    repeats=self.steps_per_epoch * self.cycle_length)

            self.lrs = self.min_lr + 0.5 * (max_lrs - self.min_lr) * (1 + np.cos(fracs * np.pi))

        else:
            num_epochs = 0
            cycle_lens = []

            while num_epochs < self.epochs:
                num_epochs += self.cycle_length
                cycle_lens.append(self.cycle_length)
                self.cycle_length = ceil(self.cycle_length * self.mult_factor)

            num_steps = num_epochs * self.steps_per_epoch
            fracs = np.zeros(num_steps)
            max_lrs = np.zeros(num_steps)

            idx = 0
            for i, l in enumerate(cycle_lens):
                steps = self.steps_per_epoch * l
                fracs[idx:steps+idx] = np.linspace(0,1,steps)
                max_lrs[idx:steps+idx] = np.repeat(self.max_lr * (self.decay ** i), steps)
                idx += steps

            self.lrs = self.min_lr + 0.5 * (max_lrs - self.min_lr) * (1 + np.cos(fracs * np.pi))


class LRFinder(Callback):

    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self, out_path):
        '''Helper function to quickly inspect the learning rate schedule.'''
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.history['iterations'], self.history['lr'])

        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning rate')

        fig.savefig(out_path)

    def plot_loss(self, out_path):
        '''Helper function to quickly observe the learning rate experiment results.'''
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.history['lr'], self.history['loss'])

        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')

        fig.savefig(out_path)
