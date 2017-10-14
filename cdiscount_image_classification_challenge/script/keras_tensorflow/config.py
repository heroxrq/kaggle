import os

from util import mkdir_if_not_exist

# --------------------------------------------------
# path config
# --------------------------------------------------
DATASET_DIR = "/home/xierenqiang/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
DEBUG = False

if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = "/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
    DEBUG = True

TRAIN_BSON_FILE = DATASET_DIR + "/train.bson"
TEST_BSON_FILE = DATASET_DIR + "/test.bson"
CATEGORY_NAMES_CSV_FILE = DATASET_DIR + "/category_names.csv"
TRAIN_EXAMPLE_BSON_FILE = DATASET_DIR + "/train_example.bson"
SAMPLE_SUBMISSION_CSV_FILE = DATASET_DIR + "/sample_submission.csv"

TRAIN_BSON_TRANSFORM_DIR = DATASET_DIR + "/train_bson_transform"
TRAIN_DIR = TRAIN_BSON_TRANSFORM_DIR + "/train"
VALID_DIR = TRAIN_BSON_TRANSFORM_DIR + "/valid"
TRAIN_RAW_DIR = TRAIN_DIR + "/raw"
VALID_RAW_DIR = VALID_DIR + "/raw"

BASE_DIR = os.path.abspath("../..")

SUBMISSION_DIR = BASE_DIR + "/submission"

MODEL_DIR = BASE_DIR + "/model"
BEST_MODEL_FILE = MODEL_DIR + '/best_model.hdf5'
BEST_WEIGHTS_FILE = MODEL_DIR + '/best_weights.hdf5'
MODEL_FILE = MODEL_DIR + "/model.json"
CLASS_INDICES_FILE = MODEL_DIR + "/class_indices.json"

LOG_DIR = BASE_DIR + "/log"
TF_LOG_DIR = LOG_DIR + "/tf_log"
PY_LOG_DIR = LOG_DIR + "/py_log"

mkdir_if_not_exist(SUBMISSION_DIR)
mkdir_if_not_exist(MODEL_DIR)
mkdir_if_not_exist(TF_LOG_DIR)

# --------------------------------------------------
# dataset config
# --------------------------------------------------
NUM_TEST_PRODUCTS = 1768182
NUM_TEST_PICS = 3095080
NUM_TRAIN_IMGS = 11134709
NUM_VALID_IMGS = 1236584

if DEBUG:
    NUM_TEST_PRODUCTS = 1000
    NUM_TEST_PICS = 1714
    NUM_TRAIN_IMGS = 14741
    NUM_VALID_IMGS = 1648

# --------------------------------------------------
# image config
# --------------------------------------------------
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180

INPUT_WIDTH = IMAGE_WIDTH
INPUT_HEIGHT = IMAGE_HEIGHT

# --------------------------------------------------
# model config
# --------------------------------------------------
SEED = 11
FC_SIZE = 8192
NUM_CLASSES = 5270

TRAIN_BATCH_SIZE = 1024
VALID_BATCH_SIZE = 1024
PREDICT_BATCH_SIZE = 1024

if DEBUG:
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 256
    PREDICT_BATCH_SIZE = 256

EPOCHS1 = 3
EPOCHS2 = 3

LR1 = 0.001
LR2 = 0.0001
