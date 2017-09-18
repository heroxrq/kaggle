import collections
import os
import random
import sys

import bson
import pandas as pd
from tqdm import tqdm_notebook

from util import mkdir_if_not_exist


def main():
    if len(sys.argv) != 5:
        print("Usage: gen_imgs_from_bson.py <validpct> <inputdir> <traindir> <validdir>")
        sys.exit(1)

    validpct = float(sys.argv[1]) / 100
    inputdir = sys.argv[2]
    traindir = sys.argv[3]
    validdir = sys.argv[4]

    trainfile = os.path.join(inputdir, 'train.bson')

    train_raw_dir = os.path.join(traindir, "raw")
    valid_raw_dir = os.path.join(validdir, "raw")

    # create categories folders
    categories = pd.read_csv(os.path.join(inputdir, 'category_names.csv'), index_col='category_id')
    for category in tqdm_notebook(categories.index):
        mkdir_if_not_exist(os.path.join(train_raw_dir, str(category)))
        mkdir_if_not_exist(os.path.join(valid_raw_dir, str(category)))

    num_products = 7069896  # 7069896 for train and 1768182 for test
    product_cnt = 0
    bar = tqdm_notebook(total=num_products)
    with open(trainfile, 'rb') as trainbson:
        data = bson.decode_file_iter(trainbson)

        train_cats, valid_cats, image_counter = collections.Counter(), collections.Counter(), collections.Counter()

        for prod in data:
            product_id = prod['_id']
            category_id = prod['category_id']

            # decide if this product will go into the validation or train data
            if random.random() < validpct:
                outdir = valid_raw_dir
                valid_cats[category_id] += 1
            else:
                outdir = train_raw_dir
                train_cats[category_id] += 1

            for picidx, pic in enumerate(prod['imgs']):
                filename = os.path.join(outdir, str(category_id), "{}.{}.jpg".format(product_id, picidx))
                with open(filename, 'wb') as f:
                    f.write(pic['picture'])

                image_counter[outdir] += 1
            bar.update()
            product_cnt += 1
            if product_cnt % 10000 == 0:
                print("converted {} products".format(product_cnt))

        for name, cnt, dir_ in [("training", train_cats, train_raw_dir), ("validation", valid_cats, valid_raw_dir)]:
            print("{}: {} categories with {} products and {} images".format(
                name, len(cnt), sum(cnt.values()), image_counter[dir_]))


if __name__ == '__main__':
    main()
