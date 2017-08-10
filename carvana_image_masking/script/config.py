DEBUG = False

IMAGE_WIDTH = 1918
IMAGE_HEIGHT = 1280

RESIZED_WIDTH = 256
RESIZED_HEIGHT = 256

BATCH_SIZE = 4
EPOCHS = 5

BASE_DIR = "/home/xrq/prog/kaggle/carvana_image_masking"

DATASET_DIR = BASE_DIR + "/dataset"

TRAIN_DIR = DATASET_DIR + "/train"
TRAIN_MASKS_DIR = DATASET_DIR + "/train_masks"
TRAIN_MASKS_CSV = DATASET_DIR + "/train_masks.csv"
TEST_DIR = DATASET_DIR + "/test"
TEST_PREDICT_DIR = DATASET_DIR + "/test_predict"

RESIZED_TRAIN_DIR = DATASET_DIR + "/resized_train"
RESIZED_TRAIN_MASKS_DIR = DATASET_DIR + "/resized_train_masks"

RESIZED_TEST_DIR = DATASET_DIR + "/resized_test"
RESIZED_TEST_PREDICT_DIR = DATASET_DIR + "/resized_test_predict"

if DEBUG:
    RESIZED_TEST_DIR = DATASET_DIR + "/mini_resized_test"
    RESIZED_TEST_PREDICT_DIR = DATASET_DIR + "/mini_resized_test_predict"

SUBMISSION_DIR = DATASET_DIR + "/submission"

TMP_DIR = BASE_DIR + "/tmp"
CHECKPOINT_DIR = TMP_DIR + "/checkpoint"
MODEL_DIR = TMP_DIR + "/model"
MODEL_FILE = MODEL_DIR + "/model.json"
WEIGHTS_FILE = MODEL_DIR + "/weights.h5"

LOG_DIR = BASE_DIR + "/log"
