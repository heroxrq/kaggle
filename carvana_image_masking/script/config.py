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

RESIZED_WIDTH = 256
RESIZED_HEIGHT = 256

if SERVER:
    RESIZED_WIDTH = 1920
    RESIZED_HEIGHT = 1280

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

SUBMISSION_DIR = DATASET_DIR + "/submission"

TMP_DIR = BASE_DIR + "/tmp"

CHECKPOINT_DIR = TMP_DIR + "/checkpoint"
BEST_WEIGHTS_FILE = CHECKPOINT_DIR + '/best_weights.hdf5'

MODEL_DIR = TMP_DIR + "/model"
MODEL_FILE = MODEL_DIR + "/model.json"
WEIGHTS_FILE = MODEL_DIR + "/weights.hdf5"

LOG_DIR = BASE_DIR + "/log"

# --------------------------------------------------
# model config
# --------------------------------------------------
TRAIN_BATCH_SIZE = 7
PREDICT_BATCH_SIZE = 40
EPOCHS = 30
LAYERS = 6
FILTERS = 32

if SERVER:
    TRAIN_BATCH_SIZE = 2
    PREDICT_BATCH_SIZE = 2
    EPOCHS = 20
    LAYERS = 8
    FILTERS = 8
