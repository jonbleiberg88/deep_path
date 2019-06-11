from utils.train_utils import split_train_test
from utils.preprocessing_utils import fix_filenames
dir = "/data/jblei/cancer_project/datasets/FL/256_0_4_processed/data"
folds = 5
if __name__ == '__main__':
    # split_train_test(dir, 5)
    fix_filenames(dir)
