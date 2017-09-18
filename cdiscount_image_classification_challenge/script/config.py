import datetime
import os

from util import mkdir_if_not_exist

DATE = datetime.datetime.now().strftime('%Y-%m-%d')
# --------------------------------------------------
# path config
# --------------------------------------------------
DATASET_DIR = "/home/xierenqiang/dataset/kaggle/cdiscount_image_classification_challenge/dataset"

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

BASE_DIR = os.path.abspath("..")

SUBMISSION_DIR = BASE_DIR + "/submission"

MODEL_DIR = BASE_DIR + "/model"
CUR_MODEL_DIR = MODEL_DIR + "/" + DATE
BEST_MODEL_FILE = CUR_MODEL_DIR + '/best_model.hdf5'
BEST_WEIGHTS_FILE = CUR_MODEL_DIR + '/best_weights.hdf5'
MODEL_FILE = CUR_MODEL_DIR + "/model.json"

LOG_DIR = BASE_DIR + "/log"
CUR_LOG_DIR = LOG_DIR + "/" + DATE
TF_LOG_DIR = CUR_LOG_DIR + "/tf_log"
PY_LOG_DIR = CUR_LOG_DIR + "/py_log"

mkdir_if_not_exist(CUR_MODEL_DIR)
mkdir_if_not_exist(TF_LOG_DIR)

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
NUM_TRAIN_IMGS = 11131442
NUM_VALID_IMGS = 1239851

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8

EPOCHS1 = 3
EPOCHS2 = 2

LR1 = 0.001
LR2 = 0.0001
