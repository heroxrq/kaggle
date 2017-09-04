#!/bin/bash

CUR_DIR=$(cd `dirname $0`; pwd)

date=$(date +%Y-%m-%d)
log_file="../log/py_log/log.$date"

#python ./dataset_util.py >> log_file 2>&1
python ./train.py >> log_file 2>&1
python ./predict.py >> log_file 2>&1
