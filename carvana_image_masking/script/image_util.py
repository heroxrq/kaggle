import os
from PIL import Image

from config import *


def image_resize(in_dir, out_dir, resized_width, resized_height, format=None, save_postfix=None):
    files = os.listdir(in_dir)
    for f in files:
        in_filename = in_dir + os.sep + f
        if save_postfix is None:
            out_filename = out_dir + os.sep + f
        else:
            out_filename = out_dir + os.sep + f.split(".")[0] + save_postfix
        with Image.open(in_filename) as img:
            resized_img = img.resize((resized_width, resized_height))
            resized_img.save(out_filename, format)


def resize_all_images():
    image_resize(TRAIN_DIR, RESIZED_TRAIN_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(TRAIN_MASKS_DIR, RESIZED_TRAIN_MASKS_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(TEST_DIR, RESIZED_TEST_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)


if __name__ == '__main__':
    # resize_all_images()
    image_resize(TRAIN_MASKS_DIR, RESIZED_TRAIN_MASKS_DIR, RESIZED_WIDTH, RESIZED_HEIGHT, 'PNG', '.png')
