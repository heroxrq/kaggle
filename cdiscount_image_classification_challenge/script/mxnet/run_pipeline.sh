#!/bin/bash

CUR_DIR=$(cd `dirname $0`; pwd)

DATASET_DIR="$HOME/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
if [ ! -d $DATASET_DIR ]; then
    DATASET_DIR="/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification_challenge/dataset"
fi
PYTHON2="$HOME/install/anaconda2/bin/python"
PYTHON3="$HOME/install/anaconda3/bin/python"

DATE=$(date +%F)

PY_LOG_DIR="../../log"
mkdir -p $PY_LOG_DIR

PY_LOG="$PY_LOG_DIR/log.$DATE"

$PYTHON3 fine_tune.py >>$PY_LOG 2>&1 &
