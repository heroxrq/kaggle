#!/bin/bash

CUR_DIR=$(cd `dirname $0`; pwd)

DATASET_DIR="/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification/dataset"
PYTHON2="/home/xrq/install/anaconda2/bin/python"
PYTHON3="/home/xrq/install/anaconda3/bin/python"

DATE=$(date +%F)

#$PYTHON3 gen_img_from_bson.py \
#10 \
#$DATASET_DIR/train.bson \
#$DATASET_DIR/train_bson_transform/train \
#$DATASET_DIR/train_bson_transform/valid

PY_LOG_DIR="../log/$DATE/py_log"
mkdir -p $PY_LOG_DIR
PY_LOG="$PY_LOG_DIR/log.$DATE"

$PYTHON2 train.py >$PY_LOG 2>&1
