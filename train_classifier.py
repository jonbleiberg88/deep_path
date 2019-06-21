import os
import tensorflow as tf
import tensorflow.keras as Keras
from data_generator import TrainDataGenerator, ValDataGenerator
from transfer_CNN import TransferCNN
from prepare_dataset import *
from learning_rate_utils import SGDRScheduler
import constants


def train_fold(folds_list, fold, data_dir=constants.PATCH_OUTPUT_DIRECTORY, epochs=constants.EPOCHS,
        model_dir=constants.MODEL_FILE_FOLDER):

    train_dict, test_dict, class_to_label = get_dataset_for_fold(data_dir, folds_list, fold)

    print("Making generators")
    train_gen = TrainDataGenerator(train_dict)
    test_gen = ValDataGenerator(test_dict)

    print("Compiling model...")
    model = TransferCNN().compile_model()
    scheduler = SGDRScheduler(min_lr=1e-5, max_lr=0.3,lr_decay=0.9, cycle_length=1)

    print("Fitting...")
    hist = model.fit_generator(train_gen, None,epochs=5,validation_data=test_gen,
                                validation_steps=None, callbacks=[scheduler])

    print("Making model dir...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving...")
    model.save(os.path.join(model_dir, f"model_fold_{fold}"))

    return hist.history['val_loss'], hist.history['val_acc']

def train_k_folds(data_dir=constants.PATCH_OUTPUT_DIRECTORY,num_folds=constants.NUM_FOLDS,
        epochs=constants.EPOCHS):

    folds_list = split_train_test(data_dir, num_folds)

    val_losses = np.zeros(num_folds)
    val_accs = np.zeros(num_folds)

    for fold in range(num_folds):
        print(f"Beginning Fold {fold}")
        val_loss, val_acc = train_fold(folds_list, fold, data_dir, epochs)

        val_losses[fold] = val_loss[-1]
        val_accs[fold] = val_acc[-1]

        print(f"Fold {fold} is complete!")


    print("Training complete!")

    return val_losses, val_accs

if __name__ == "__main__":
    val_losses, val_accs = train_k_folds()

    print(f"Validation Loss: Mean: {np.mean(val_losses):.2f}, Max: {np.max(val_losses):.2f}, Min: {np.min(val_losses):.2f}")
    print(f"Validation Accuracy: Mean: {np.mean(val_accs)*100:.2f}, Max: {np.max(val_accs)*100:.2f}, Min: {np.min(val_accs)*100:.2f}")
