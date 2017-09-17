# https://www.kaggle.com/nimararora/script-to-generate-raw-images-in-subdirs
# Creates subdirectories with raw images to use for training with keras.preprocessing.image.ImageDataGenerator.
# 1. mkdir input
# 2. download train.bson to input/
# 3. mkdir train_bson_transform
# 4. create a file called gen_img_from_bson.py with the contents of this script
# 5. python3 gen_img_from_bson.py 10 input/train.bson train_bson_transform/train train_bson_transform/valid
# 6. wait a day
# 7. ls train_bson_transform/train/raw train_bson_transform/valid/raw
import collections
import io
import os
import random
import sys

import bson
import scipy
import skimage.data

from util import mkdir_if_not_exist


def main():
    if len(sys.argv) != 5:
        print("Usage: gen_img_from_bson.py <valid-pct> <bson-file> <traindir> <validdir>")
        sys.exit(1)

    validpct = float(sys.argv[1]) / 100
    trainfile = sys.argv[2]
    traindir = sys.argv[3]
    validdir = sys.argv[4]

    for dir_ in [traindir, validdir]:
        mkdir_if_not_exist(dir_)
        mkdir_if_not_exist(os.path.join(dir_, "raw"))

    data = bson.decode_file_iter(open(trainfile, 'rb'))

    train_cats, valid_cats = collections.Counter(), collections.Counter()

    img_cnt = 0
    product_cnt = 0

    for prod in data:
        product_id = prod['_id']
        category_id = prod['category_id']

        # decide if this product will go into the validation or train data
        if random.random() < validpct:
            outdir = validdir
            valid_cats[category_id] += 1
        else:
            outdir = traindir
            train_cats[category_id] += 1

        cat_dir = os.path.join(outdir, "raw", str(category_id))
        mkdir_if_not_exist(cat_dir)

        for picidx, pic in enumerate(prod['imgs']):
            picture = skimage.data.imread(io.BytesIO(pic['picture']))

            filename = os.path.join(cat_dir, "{}.{}.jpg".format(product_id, picidx))
            scipy.misc.imsave(filename, picture)

            img_cnt += 1
        product_cnt += 1
        if product_cnt % 10000 == 0:
            print("converted {} products, {} images".format(product_cnt, img_cnt))

    for name, cnt in [("training", train_cats), ("validation", valid_cats)]:
        print("{}: {} categories with {} products".format(name, len(cnt), sum(cnt.values())))


if __name__ == '__main__':
    main()
