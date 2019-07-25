import os
import tensorflow as tf
import tensorflow.keras as Keras
from data_generator import TrainDataGenerator, ValDataGenerator
from transfer_CNN import TransferCNN
from prepare_dataset import *
from learning_rate_utils import SGDRScheduler
from utils.file_utils import write_pickle_to_disk
import constants


def train_fold(folds_list, fold, class_to_label, data_dir=constants.PATCH_OUTPUT_DIRECTORY, epochs=constants.EPOCHS,
        model_dir=constants.MODEL_FILE_FOLDER, class_counts=None):

    train_dict, test_dict = get_dataset_for_fold(data_dir, folds_list, fold, class_to_label)

    if class_counts is not None and constants.LOSS_WEIGHTING:
        fold_counts = class_counts[fold]['train']
        class_weights = {}
        total_num = sum(list(fold_counts.values()))
        for class_name, num in fold_counts.items():
            class_weights[class_to_label[class_name]] = (1 / num) / (1/ num + 1/(total_num - num))
        print(class_weights)
    else:
        class_weights = None
        print("Weights are None")

    print("Making generators")
    train_gen = TrainDataGenerator(train_dict)
    test_gen = ValDataGenerator(test_dict)

    print("Compiling model...")
    model, base_model = TransferCNN().compile_model()
    if fold == 0:
        print(model.summary())


    print("Fitting...")
    if constants.USE_SGDR:
        scheduler = SGDRScheduler(min_lr=constants.MIN_LR, max_lr=constants.MAX_LR,
                                    lr_decay=constants.LR_DECAY, cycle_length=constants.CYCLE_LENGTH,
                                    mult_factor=constants.CYCLE_MULT)
        hist = model.fit_generator(train_gen, None,epochs=epochs,validation_data=test_gen,
                                    validation_steps=None, callbacks=[scheduler], class_weight=class_weights)
    else:
        hist = model.fit_generator(train_gen, None,epochs=epochs,validation_data=test_gen,
                                    validation_steps=None, class_weight=class_weights)

    print("Making model dir...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving...")
    base_model.save(os.path.join(model_dir, f"model_fold_{fold}"))

    return hist.history['val_loss'], hist.history['val_acc']

def train_k_folds(data_dir=constants.PATCH_OUTPUT_DIRECTORY,num_folds=constants.NUM_FOLDS,
        epochs=constants.EPOCHS):

    folds_list, class_to_label, class_counts = split_train_test(data_dir, num_folds)

    val_losses = np.zeros(num_folds)
    val_accs = np.zeros(num_folds)

    for fold in range(num_folds):
        print(f"Beginning Fold {fold} of {num_folds - 1}")
        val_loss, val_acc = train_fold(folds_list, fold, class_to_label, data_dir, epochs, class_counts=class_counts)

        val_losses[fold] = val_loss[-1]
        val_accs[fold] = val_acc[-1]

        print(f"Fold {fold} of {num_folds - 1} is complete!")


    print("Training complete!")

    return val_losses, val_accs


if __name__ == "__main__":
    val_losses, val_accs = train_k_folds()

    print(f"Validation Loss: Mean: {np.mean(val_losses):.2f}, Median: {np.median(val_losses):.2f}, Max: {np.max(val_losses):.2f}, Min: {np.min(val_losses):.2f}")
    print(f"Validation Accuracy: Mean: {np.mean(val_accs)*100:.2f}, Median: {np.median(val_accs)*100:.2f}, Max: {np.max(val_accs)*100:.2f}, Min: {np.min(val_accs)*100:.2f}")
