import os

# Training Parameters
NUM_FOLDS = 5
EPOCHS = 6
BATCH_SIZE = 64
BATCHES_PER_EPOCH = 200

OPTIMIZER = 'adam'
LEARNING_RATE = 0.01

# SGDR Parameters
USE_SGDR = True
MIN_LR = 1e-6
MAX_LR = 0.03
LR_DECAY = 0.5
CYCLE_LENGTH = 3
CYCLE_MULT = 2


GPUS = 2

# Dataset Parameters
DEFAULT_CLASS_NAME = "normal_tissue"

HISTOGRAM_THRESHOLD = True
REMOVE_BLANK_TILES = True
BLACK_WHITE_THRESHOLD = 0.9
REMOVE_THRESHOLD = 0.9
DEFAULT_CLASS_REMOVE_THRESHOLD = 0.5
BLUR_RADIUS = 7
EXPORT_RESULTS = False
EXPORT_DIR = None

NUM_CLASSES = 4


STRATIFY = False


BALANCE_CLASSES = True
WEIGHT_BY_SIZE = True



USE_AUGMENTATION = True
# Augmentation Parameters
ROTATION_RANGE = 0
WIDTH_SHIFT_RANGE = 0.0
HEIGHT_SHIFT_RANGE = 0.0
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ZOOM_RANGE = 0.0
FILL_MODE = 'reflect'

RESIZE_IMAGES = False
INPUT_IMAGE_DIM = (128, 128)
OUTPUT_IMAGE_DIM = (128, 128)
N_CHANNELS = 3

SLIDE_FILE_EXTENSION     = ("svs","qptiff")
OVERLAP                  = 0
PATCH_SIZE               = INPUT_IMAGE_DIM[0] - (OVERLAP * 2)
NUM_VERTICES_IN_ANNOTATION = 3

USE_TTA = True
TTA_AUG_TIMES = 3

# Model Parameters
BASE_ARCHITECTURE = 'ResNet50'
OUTPUT_POOLING = 'avg'
INPUT_SHAPE = (*OUTPUT_IMAGE_DIM, N_CHANNELS)
FREEZE = False

LAYER_SIZES = [1024, 256]
USE_BATCH_NORM = True
USE_DROPOUT = True
DROPOUT_RATE = 0.5



METRICS = ['accuracy']

# Post Processing Parameters
KNN_SMOOTH = True
KNN_RANGE = 1
SMOOTH_FACTOR = 0.7

# Data directories

MODE = 'remote'
# MODE ='jupyter'
# MODE = 'local'

DATASET = 'FL'
# DATASET = 'CLL'

if MODE == 'remote':
    if DATASET == 'FL':
        SLIDE_FILE_DIRECTORY     = "/dp/datasets/FL/raw_slides/slide_imgs"
        OUTPUT_DIRECTORY = f"/dp/datasets/FL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_multiclass/"
        ANNOTATION_CSV_DIRECTORY = "/dp/datasets/FL/raw_slides/annotations/"
        SA_CSV_DIRECTORY = "/dp/datasets/FL/raw_slides/surface_areas/"
        MODEL_FILE_FOLDER       = "/dp/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP           = ['FLN02_Scan1.qptiff', 'FLN04_Scan1.qptiff']

    elif DATASET == 'CLL':
        SLIDE_FILE_DIRECTORY     = "/dp/datasets/CLL/raw_slides/slide_imgs"
        OUTPUT_DIRECTORY = f"/dp/datasets/CLL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_multiclass/"
        ANNOTATION_CSV_DIRECTORY = "/dp/datasets/CLL/raw_slides/annotations/"
        MODEL_FILE_FOLDER       = "/dp/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP         = ['CLT10_Scan3.qptiff', 'CLN17_Scan1.qptiff', 'CLN28_Scan1.qptiff']

if MODE == 'jupyter':
    if DATASET == 'FL':
        SLIDE_FILE_DIRECTORY     = "/tf/dp/datasets/FL/raw_slides/slide_imgs"
        OUTPUT_DIRECTORY = f"/tf/dp/datasets/FL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_multiclass/"
        ANNOTATION_CSV_DIRECTORY = "/tf/dp/datasets/FL/raw_slides/annotations/"
        SA_CSV_DIRECTORY = "/tf/dp/datasets/FL/raw_slides/surface_areas/"
        MODEL_FILE_FOLDER       = "/tf/dp/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP           = ['FLN02_Scan1.qptiff', 'FLN04_Scan1.qptiff']

    elif DATASET == 'CLL':
        SLIDE_FILE_DIRECTORY     = "/tf/dp/datasets/CLL/raw_slides/slide_imgs"
        OUTPUT_DIRECTORY = f"/tf/dp/datasets/CLL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_multiclass/"
        ANNOTATION_CSV_DIRECTORY = "/tf/dp/datasets/CLL/raw_slides/annotations/"
        MODEL_FILE_FOLDER       = "/tf/dp/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP         = ['CLT10_Scan3.qptiff', 'CLN17_Scan1.qptiff', 'CLN28_Scan1.qptiff']

elif MODE == 'local':
    if DATASET == 'FL':
        SLIDE_FILE_DIRECTORY     = "/Volumes/Backup/Projects/cancer_project/5:21:19/FL Scans"
        OUTPUT_DIRECTORY   = f"/Volumes/Backup/Projects/cancer_project/datasets/FL/test/"
        ANNOTATION_CSV_DIRECTORY = "/Volumes/Backup/Projects/cancer_project/5:21:19/FL_Proj/annotation_csv_files/"
        MODEL_FILE_FOLDER       = "//Users/jonathanbleiberg/Documents/College/Research/cancer_project/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP           = ['FLN02_Scan1.qptiff', 'FLN04_Scan1.qptiff']

        # QuPath constants
        ANNOTATION_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/FL Annotations"
        IMAGE_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/FL Scans"
        PROJECT_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/FL_Proj"
        PROJECT_FILE = "project.qpproj"

    elif DATASET == 'CLL':
        SLIDE_FILE_DIRECTORY     = "/Volumes/Backup/Projects/cancer_project/5:21:19/CLL Scans"
        OUTPUT_DIRECTORY = f"/Volumes/Backup/Projects/cancer_project/5:21:19/Datasets/CLL/test/"
        ANNOTATION_CSV_DIRECTORY = "/Volumes/Backup/Projects/cancer_project/5:21:19/CLL_Proj2/annotation_csv_files/"
        MODEL_FILE_FOLDER       = "//Users/jonathanbleiberg/Documents/College/Research/cancer_project/models/output_graph_files_" + str(PATCH_SIZE)
        FILES_TO_SKIP         = ['CLT10_Scan3.qptiff', 'CLN17_Scan1.qptiff', 'CLN28_Scan1.qptiff']

        # QuPath constants
        ANNOTATION_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/CLL Annotations"
        IMAGE_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/CLL Scans"
        PROJECT_DIR = "/Volumes/Backup/Projects/cancer_project/5:21:19/CLL_Proj2"
        PROJECT_FILE = "project.qpproj"

PATCH_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "data")
PREDICTIONS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "predictions")
HELPER_FILES_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "helper_files")
LARGE_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "large_tumor")
SMALL_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "small_tumor")

#LABEL_FILE_PATH         = "/data/ethan/Breast_Deep_Learning/labels.csv"
#LABEL_FILE               = "/data/ethan/lymphoma_case_codes.csv"




#Constants for pre-trained models

MODEL_FILE_FOLDER   = f"/dp/models/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_multiclass/"


#Visualization helper files
VISUALIZATION_HELPER_FILE_FOLDER = os.path.join(HELPER_FILES_DIRECTORY, "visualization_helper_files")

PATCH_CONFIDENCE_FOLDER          = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_confidences")
PATCH_NAME_TO_COORDS_MAP         = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_name_to_coords_map")
SLIDE_NAME_TO_TILE_DIMS_MAP      = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_tile_dims_map")
SLIDE_NAME_TO_PATCHES_MAP        = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_patches_map")
