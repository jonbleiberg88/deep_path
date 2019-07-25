import os
import tensorflow as tf
import tensorflow.keras as Keras
import pandas as pd
from data_generator import TrainDataGenerator, ValDataGenerator, TestDataGenerator
from transfer_CNN import TransferCNN
from prepare_dataset import *
from learning_rate_utils import SGDRScheduler
from utils.file_utils import write_pickle_to_disk
import constants

def create_leave_one_out_lists(data_dir=constants.PATCH_OUTPUT_DIRECTORY):
    img_list = []
    classes = os.listdir(data_dir)
    class_to_label = {c:i for i,c in enumerate(classes)}

    for class_dir in classes:
        full_path = os.path.join(data_dir, class_dir)
        img_list += os.listdir(full_path)
    img_list = set(img_list)

    folds_list = [{'train':[], 'test': []} for _ in range(len(img_list))]

    for idx, img in enumerate(img_list):
        folds_list[idx]['train'] = list(img_list - set([img]))
        folds_list[idx]['test'] = [img]

    return folds_list, class_to_label


def train_and_predict_fold(folds_list, fold, class_to_label, data_dir=constants.PATCH_OUTPUT_DIRECTORY, epochs=constants.EPOCHS,
        model_dir=constants.MODEL_FILE_FOLDER, predict_dir=constants.PREDICTIONS_DIRECTORY, show_val=False):

    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    train_dict, test_dict = get_dataset_for_fold(data_dir, folds_list, fold, class_to_label)

    if len(test_dict.keys()) == 0:
        print(f"No image files found for fold {fold}")
        return -1, -1

    predict_slide = folds_list[fold]['test'][0]

    print("Making generators...")
    train_gen = TrainDataGenerator(train_dict)
    if show_val:
        test_gen = ValDataGenerator(test_dict)

    predict_gen = TestDataGenerator(test_dict)

    print("Compiling model...")
    model = TransferCNN().compile_model()
    if fold == 0:
        print(model.summary())
    if constants.USE_SGDR:
        scheduler = SGDRScheduler(min_lr=constants.MIN_LR, max_lr=constants.MAX_LR,
                                    lr_decay=constants.LR_DECAY, cycle_length=constants.CYCLE_LENGTH,
                                    mult_factor=constants.CYCLE_MULT)
    else:
        scheduler = None

    print("Fitting...")
    if show_val:
        model.fit_generator(train_gen, None,epochs=epochs,validation_data=test_gen,
                                validation_steps=None, callbacks=[scheduler])
    else:
        model.fit_generator(train_gen, None,epochs=epochs, callbacks=[scheduler])

    print(f"Predicting {predict_slide}...")
    preds = model.predict_generator(predict_gen, None, verbose=1)

    paths, preds = predict_gen.get_predictions(preds)

    loss, accuracy = predict_gen.eval(preds)

    print(f"Test Loss: {loss:.2f}; Test Accuracy: {accuracy*100:.2f}%")

    print(f"Saving predictions...")

    preds_df = pd.DataFrame({'filepath': paths, 'labels':predict_gen.get_labels()})
    preds_df = pd.concat([preds_df, pd.DataFrame(preds)], axis=1)
    preds_df.to_csv(f"{predict_dir}/{predict_slide}.csv")


    print("Saving model...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(os.path.join(model_dir, f"weights_fold_{fold}"))

    return loss, accuracy

def train_and_predict_all(data_dir=constants.PATCH_OUTPUT_DIRECTORY,
        epochs=constants.EPOCHS, model_dir=constants.MODEL_FILE_FOLDER):

    folds_list, class_to_label = create_leave_one_out_lists(data_dir)
    write_pickle_to_disk(f'{constants.VISUALIZATION_HELPER_FILE_FOLDER}/class_to_label',
                            class_to_label)
    num_slides = len(folds_list)

    losses = np.zeros(num_slides)
    accs = np.zeros(num_slides)
    empty = []

    for fold in range(num_slides):
        print(f"Beginning Fold {fold} of {num_slides - 1}")
        loss, acc = train_and_predict_fold(folds_list, fold, class_to_label, data_dir, epochs)
        if loss == -1 and acc == -1:
            empty.append(fold)
        else:
            losses[fold] = loss
            accs[fold] = acc

        print(f"Fold {fold} of {num_slides - 1} is complete!")

    losses = np.delete(losses, empty)
    accs = np.delete(accs, empty)

    print("Training model on full dataset...")
    slides = folds_list[0]['train'] + folds_list[0]['test']

    train_dict = get_full_dataset(data_dir, slides, class_to_label)
    train_gen = TrainDataGenerator(train_dict)

    model = TransferCNN().compile_model()
    if constants.USE_SGDR:
        scheduler = SGDRScheduler(min_lr=constants.MIN_LR, max_lr=constants.MAX_LR,
                                    lr_decay=constants.LR_DECAY, cycle_length=constants.CYCLE_LENGTH,
                                    mult_factor=constants.CYCLE_MULT)
    else:
        scheduler = None

    print("Fitting...")
    model.fit_generator(train_gen, None,epochs=epochs, callbacks=[scheduler])

    print("Saving final model...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, f"final_model"))

    print("Training and prediction complete!")
    return losses, accs

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    losses, accs = train_and_predict_all()

    print(f"Test Loss: Mean: {np.mean(losses):.2f}, Median: {np.median(losses):.2f}, Max: {np.max(losses):.2f}, Min: {np.min(losses):.2f}")
    print(f"Test Accuracy: Mean: {np.mean(accs)*100:.2f}%, Median: {np.median(accs)*100:.2f}%,Max: {np.max(accs)*100:.2f}%, Min: {np.min(accs)*100:.2f}%")
