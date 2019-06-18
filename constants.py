import os

#Training Parameters
NUM_FOLDS = 3
EPOCHS = 5

#Data directories
# SLIDE_FILE_DIRECTORY     = "/Volumes/Elements/5:21:19/FL Scans"
# SLIDE_FILE_DIRECTORY     = "/Volumes/Elements/5:21:19/CLL Scans"
SLIDE_FILE_DIRECTORY     = "/dp/datasets/FL/raw_slides/slide_imgs"

SLIDE_FILE_EXTENSION     = ("svs","qptiff")
OVERLAP                  = 0
PATCH_SIZE               = 256 - (OVERLAP * 2)
NUM_VERTICES_IN_ANNOTATION = 4


# OUTPUT_DIRECTORY   = f"/Volumes/Elements/5:21:19/Datasets/FL/test3/"
# PATCH_OUTPUT_DIRECTORY   = f"/Volumes/Elements/5:21:19/Datasets/CLL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}_processed/"
# PATCH_OUTPUT_DIRECTORY = "/data/jblei/cancer_project/datasets/FL/256_0_4_processed/data/"
OUTPUT_DIRECTORY = f"/dp/datasets/FL/{str(PATCH_SIZE)}_{str(OVERLAP)}_{str(NUM_VERTICES_IN_ANNOTATION)}/"
PATCH_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "data")
LARGE_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "large_tumor_cells")
SMALL_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "small_tumor_cells")
#LABEL_FILE_PATH         = "/data/ethan/Breast_Deep_Learning/labels.csv"
#LABEL_FILE               = "/data/ethan/lymphoma_case_codes.csv"
# ANNOTATION_CSV_DIRECTORY = "/Volumes/Elements/5:21:19/FL_Proj/annotation_csv_files/"
# ANNOTATION_CSV_DIRECTORY = "/Volumes/Elements/5:21:19/CLL_Proj2/annotation_csv_files/"
ANNOTATION_CSV_DIRECTORY = "/dp/datasets/FL/raw_slides/annotations/"
#Constants for pre-trained models
HOW_MANY_TRAINING_STEPS = 50

BOTTLENECK_DIR          = "/dp/.tmp/bottleneck_" + str(PATCH_SIZE)
MODEL_FILE_FOLDER       = "/dp/models/output_graph_files_" + str(PATCH_SIZE)
INPUT_LAYER             = "Placeholder"
OUTPUT_LAYER            = "final_result"
TEST_SLIDE_FOLDER       = "/dp/models/testing_slide_lists_" + str(PATCH_SIZE)
TEST_SLIDE_LIST         = "testing_slide_list"

# FILES_TO_SKIP         = ['CLT10_Scan3.qptiff', 'CLN17_Scan1.qptiff', 'CLN28_Scan1.qptiff']
FILES_TO_SKIP           = ['FLN02_Scan1.qptiff', 'FLN04_Scan1.qptiff']
#Visualization output locations
HISTOGRAM_FOLDER = "histograms"
def HISTOGRAM_SUBFOLDER(fold_number):
    return os.path.join(HISTOGRAM_FOLDER, "fold_" + str(fold_number))
HEATMAP_FOLDER = "heatmaps"
def HEATMAP_SUBFOLDER(fold_number):
    return os.path.join(HEATMAP_FOLDER, "fold_" + str(fold_number))

#Visualization helper files
#VISUALIZATION_HELPER_FILE_FOLDER = os.path.join("/data/jblei/cancer_project/datasets/FL/256_0_4_processed/data/","visualization_helper_files_" + str(PATCH_SIZE))
VISUALIZATION_HELPER_FILE_FOLDER = os.path.join(OUTPUT_DIRECTORY, "visualization_helper_files")

PATCH_CONFIDENCE_FOLDER          = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_confidences")
PATCH_NAME_TO_COORDS_MAP         = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_name_to_coords_map")
SLIDE_NAME_TO_TILE_DIMS_MAP      = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_tile_dims_map")
SLIDE_NAME_TO_PATCHES_MAP        = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_patches_map")
FOLD_VOTE_CONTAINER_LISTS_PATH   = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "fold_vote_container_lists")

def PATCH_CONFIDENCE_FOLD_SUBFOLDER(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number))
def PATCH_NAME_TO_CONFIDENCE_MAP(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "patch_name_to_confidence_map")
def CONFIDENCE_CONTAINER_LIST(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "confidence_containers")
def POS_SLIDE_CONFIDENCE_LISTS(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "pos_slide_confidence_lists")
def NEG_SLIDE_CONFIDENCE_LISTS(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "neg_slide_confidence_lists")
