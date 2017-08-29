import os

# --------------------------------------------------
# env config
# --------------------------------------------------
DEBUG = False
SERVER = False

# --------------------------------------------------
# image config
# --------------------------------------------------
IMAGE_WIDTH = 1918
IMAGE_HEIGHT = 1280

RESIZED_WIDTH = 1920
RESIZED_HEIGHT = 1280

INPUT_WIDTH = RESIZED_WIDTH
INPUT_HEIGHT = RESIZED_HEIGHT

OUTPUT_WIDTH = IMAGE_WIDTH
OUTPUT_HEIGHT = IMAGE_HEIGHT

# --------------------------------------------------
# path config
# --------------------------------------------------
BASE_DIR = os.path.abspath("..")

DATASET_DIR = BASE_DIR + "/dataset"

TRAIN_DIR = DATASET_DIR + "/train"
TRAIN_MASKS_DIR = DATASET_DIR + "/train_masks"
TRAIN_MASKS_CSV = DATASET_DIR + "/train_masks.csv"
TEST_DIR = DATASET_DIR + "/test"

RESIZED_TRAIN_DIR = DATASET_DIR + "/resized_train"
RESIZED_TRAIN_MASKS_DIR = DATASET_DIR + "/resized_train_masks"
RESIZED_TEST_DIR = DATASET_DIR + "/resized_test"

INPUT_TRAIN_DIR = RESIZED_TRAIN_DIR
INPUT_TRAIN_MASKS_DIR = TRAIN_MASKS_DIR
INPUT_TEST_DIR = RESIZED_TEST_DIR

SUBMISSION_DIR = DATASET_DIR + "/submission"

MODEL_DIR = BASE_DIR + "/model"

BEST_WEIGHTS_FILE = MODEL_DIR + '/best_weights.hdf5'
MODEL_FILE = MODEL_DIR + "/model.json"

LOG_DIR = BASE_DIR + "/log"
TF_LOG_DIR = LOG_DIR + "/tf_log"
PY_LOG_DIR = LOG_DIR + "/py_log"

# --------------------------------------------------
# model config
# --------------------------------------------------
TRAIN_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
EPOCHS = 1
LAYERS = 8
FILTERS = 4

if SERVER:
    TRAIN_BATCH_SIZE = 3
    PREDICT_BATCH_SIZE = 10
    EPOCHS = 50
    LAYERS = 8
    FILTERS = 8
