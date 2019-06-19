import os
import tensorflow as tf
import tensorflow.keras as Keras
from data_generator import TrainDataGenerator, ValDataGenerator
from transfer_CNN import TransferCNN
from prepare_dataset import *
import constants

def train_fold(folds_list, fold, data_dir=constants.PATCH_OUTPUT_DIRECTORY, epochs=constants.EPOCHS,
        model_dir=constants.MODEL_FILE_FOLDER):

    train_dict, test_dict, class_to_label = get_dataset_for_fold(data_dir, folds_list, fold)

    print("Making generators")
    train_gen = TrainDataGenerator(train_dict)
    test_gen = ValDataGenerator(test_dict)

    print("Compiling model...")
    model = TransferCNN().compile_model()

    print("Fitting...")
    model.fit_generator(train_gen, None,epochs=5,validation_data=test_gen, validation_steps=None)

    print("Making model dir...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving...")
    model.save(os.path.join(model_dir, f"model_fold_{fold}"))


def train_k_folds(data_dir=constants.PATCH_OUTPUT_DIRECTORY,num_folds=constants.NUM_FOLDS,
        epochs=constants.EPOCHS):

    folds_list = split_train_test(data_dir, num_folds)

    for fold in range(num_folds):
        print(f"Beginning Fold {fold}")
        train_fold(folds_list, fold, data_dir, epochs)
        print(f"Fold {fold} is complete!")


    print("Training complete!")

    return

if __name__ == "__main__":
    data_dir=constants.PATCH_OUTPUT_DIRECTORY
    folds_list = split_train_test(data_dir, 2)
    train_fold(folds_list, 0, data_dir, 5)
