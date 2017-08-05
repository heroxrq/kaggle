import numpy as np
import os
from PIL import Image

from config import *


def image_resize(in_dir, out_dir, resized_width, resized_height):
    files = os.listdir(in_dir)
    for f in files:
        in_filename = in_dir + os.sep + f
        out_filename = out_dir + os.sep + f
        with Image.open(in_filename) as img:
            resized_img = img.resize((resized_width, resized_height))

            if not DEBUG:
                resized_img.save(out_filename)
            else:
                # print image info
                print "-----origin-----"
                print img.getbands()
                print img.mode
                print img.size
                pic = np.array(img, np.float32)
                print pic.max()
                print pic.min()
                print pic.shape
                print type(pic)

                print "-----resized-----"
                print resized_img.getbands()
                print resized_img.mode
                print resized_img.size
                pic = np.array(resized_img, np.float32)
                print pic.max()
                print pic.min()
                print pic.shape
                print type(pic)
                resized_img.show()
                return 0


if __name__ == '__main__':
    image_resize(TRAIN_DIR, RESIZED_TRAIN_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(TRAIN_MASKS_DIR, RESIZED_TRAIN_MASKS_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(TEST_DIR, RESIZED_TEST_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(RESIZED_TEST_PREDICT_DIR, TEST_PREDICT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
