#!/bin/bash

CUR_DIR=$(cd `dirname $0`; pwd)

DATASET_DIR="$HOME/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
if [ -d $DATASET_DIR ]; then
    DATASET_DIR="/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
fi
PYTHON2="$HOME/install/anaconda2/bin/python"
PYTHON3="$HOME/install/anaconda3/bin/python"

DATE=$(date +%F)

PY_LOG_DIR="../log/$DATE/py_log"
mkdir -p $PY_LOG_DIR

PY_LOG="$PY_LOG_DIR/log.$DATE"

# This is only needed for the first time.
#$PYTHON2 gen_imgs_from_bson.py \
#10 \
#$DATASET_DIR \
#$DATASET_DIR/train_bson_transform/train \
#$DATASET_DIR/train_bson_transform/valid

$PYTHON2 train.py >>$PY_LOG 2>&1
$PYTHON2 predict.py >>$PY_LOG 2>&1
