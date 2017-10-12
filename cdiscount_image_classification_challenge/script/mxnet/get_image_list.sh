py3 /home/xierenqiang/install/incubator-mxnet/tools/im2rec.py --list True --recursive True --train-ratio 0.9 train_all /home/xierenqiang/dataset/kaggle/cdiscount_image_classification_challenge/dataset/train_bson_transform/train_all/raw

py3 /home/xierenqiang/install/incubator-mxnet/tools/im2rec.py --quality 95 --num-thread $(nproc) train_all /home/xierenqiang/dataset/kaggle/cdiscount_image_classification_challenge/dataset/train_bson_transform/train_all/raw
