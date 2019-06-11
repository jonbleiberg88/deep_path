from utils.train_utils import split_train_test
from utils.preprocessing_utils import fix_filenames
dir = "/Volumes/Elements/5:21:19/Datasets/FL/test/data/"
folds = 5
if __name__ == '__main__':
    fix_filenames(dir)
