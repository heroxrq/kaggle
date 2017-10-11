#!/bin/bash

$HOME/install/anaconda3/bin/python \
$HOME/install/anaconda3/lib/python3.6/site-packages/mxnet/tools/im2rec.py \
--list True --recursive True --train-ratio 1.0 train_raw \
/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification_challenge/dataset/train_bson_transform/train/raw

$HOME/install/anaconda3/bin/python \
$HOME/install/anaconda3/lib/python3.6/site-packages/mxnet/tools/im2rec.py \
--list True --recursive True --train-ratio 1.0 valid_raw \
/media/xrq/Elements/dataset/kaggle/cdiscount_image_classification_challenge/dataset/train_bson_transform/valid/raw
