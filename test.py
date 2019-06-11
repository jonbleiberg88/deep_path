from utils.train_utils import split_train_test

dir = "/Volumes/Elements/5:21:19/Datasets/FL/test/data/"
folds = 5
if __name__ == '__main__':
    split_train_test(dir, folds)
