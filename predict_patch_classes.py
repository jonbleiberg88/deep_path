import os
import tensorflow as tf
import tensorflow.keras as Keras
import pandas as pd
from data_generator import TrainDataGenerator, ValDataGenerator, TestDataGenerator
from transfer_CNN import TransferCNN
from prepare_dataset import *
from learning_rate_utils import SGDRScheduler
import constants

def create_leave_one_out_lists(data_dir=constants.PATCH_OUTPUT_DIRECTORY):
    img_list = []
    for class_dir in os.listdir(data_dir):
        full_path = os.path.join(data_dir, class_dir)
        img_list += os.listdir(full_path)
    img_list = set(img_list)

    folds_list = [{'train':[], 'test': []} for _ in range(len(img_list))]

    for idx, img in enumerate(img_list):
        folds_list[idx]['train'] = list(img_list - set([img]))
        folds_list[idx]['test'] = [img]

    return folds_list

def train_and_predict_fold(folds_list, fold, data_dir=constants.PATCH_OUTPUT_DIRECTORY, epochs=constants.EPOCHS,
        model_dir=constants.MODEL_FILE_FOLDER, predict_dir=constants.PREDICTIONS_DIRECTORY):

    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    train_dict, test_dict, class_to_label = get_dataset_for_fold(data_dir, folds_list, fold)
    print(test_dict)
    if len(test_dict.keys()) == 0:
        print(f"No image files found for fold {fold}")
        return -1, -1

    predict_slide = list(test_dict.keys())[0]

    print("Making generators...")
    train_gen = TrainDataGenerator(train_dict)
    test_gen = ValDataGenerator(test_dict)
    predict_gen = TestDataGenerator(test_dict)

    print("Compiling model...")
    model = TransferCNN().compile_model()
    if fold == 0:
        print(model.summary())
    scheduler = SGDRScheduler(min_lr=1e-6, max_lr=0.01,lr_decay=0.5, cycle_length=2)

    print("Fitting...")
    model.fit_generator(train_gen, None,epochs=epochs,validation_data=test_gen,
                                validation_steps=None, callbacks=[scheduler])

    print(f"Predicting {predict_slide}...")
    preds = model.predict_generator(predict_gen, None)
    if predict_gen.use_tta:
        paths, preds = predict_gen.extract_TTA_preds(preds)
    else:
        paths = predict_gen.paths()

    loss, accuracy = predict_gen.eval(preds)
    print(f"Test Loss: {loss:.2f}; Test Accuracy: {accuracy:.2f}")

    print(f"Saving predictions...")
    preds_df = pd.DataFrame({'filepath': paths, 'prediction': preds})
    preds_df.to_csv(f"{predict_dir}/{predict_slide}.csv")


    print("Saving model...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, f"model_fold_{fold}"))

    return loss, accuracy

def train_and_predict_all(data_dir=constants.PATCH_OUTPUT_DIRECTORY,
        epochs=constants.EPOCHS):

    folds_list = create_leave_one_out_lists(data_dir)
    num_slides = len(folds_list)

    losses = np.zeros(num_slides)
    accs = np.zeros(num_slides)
    empty = []

    for fold in range(num_slides):
        print(f"Beginning Fold {fold}")
        loss, acc = train_and_predict_fold(folds_list, fold, data_dir, epochs)
        if loss == -1 and acc == -1:
            empty.append(fold)
        else:
            losses[fold] = loss
            accs[fold] = acc

        print(f"Fold {fold} is complete!")

    losses = np.delete(losses, empty)
    accs = np.delete(accs, empty)

    print("Training and prediction complete!")

    return losses, accs

if __name__ == "__main__":
    losses, accs = train_and_predict_all()

    print(f"Test Loss: Mean: {np.mean(losses):.2f}, Median: {np.median(losses):.2f}, Max: {np.max(losses):.2f}, Min: {np.min(losses):.2f}")
    print(f"Test Accuracy: Mean: {np.mean(accs)*100:.2f}, Median: {np.median(accs)*100:.2f},Max: {np.max(accs)*100:.2f}, Min: {np.min(accs)*100:.2f}")
