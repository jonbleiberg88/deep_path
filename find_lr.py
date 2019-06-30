from learning_rate_utils import *
from transfer_CNN import TransferCNN
from data_generator import TrainDataGenerator
from prepare_dataset import *
import constants

def find_learning_rate(data_dir=constants.PATCH_OUTPUT_DIRECTORY, num_folds=constants.NUM_FOLDS,
                        epochs=3):
    folds_list = split_train_test(data_dir, num_folds)
    train_dict, _, _ = get_dataset_for_fold(data_dir, folds_list, 0)

    data_gen = TrainDataGenerator(train_dict)

    lr_finder = LRFinder(min_lr=1e-7,
                         max_lr=3,
                         steps_per_epoch=constants.BATCHES_PER_EPOCH,
                         epochs=epochs)
    # with tf.device('/cpu:0'):
    model = TransferCNN().compile_model()

    model.fit_generator(data_gen, None, epochs=epochs, callbacks=[lr_finder])

    lr_finder.plot_lr(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/lr.png')
    lr_finder.plot_loss(f"{constants.VISUALIZATION_HELPER_FILE_FOLDER}/loss.png")


if __name__ == '__main__':
    find_learning_rate()
