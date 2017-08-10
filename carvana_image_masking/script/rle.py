import numpy as np
import pandas as pd
from scipy import ndimage

from config import *


def run_length_encoding(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join(str(x) for x in runs)
    return rle


def test_run_length_encoding():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
    print run_length_encoding(test_mask)
    assert run_length_encoding(test_mask) == '7 2 11 3'

    train_masks = pd.read_csv(TRAIN_MASKS_CSV)
    for i in xrange(len(train_masks)):
        name = train_masks['img'][i]
        rle_mask = train_masks['rle_mask'][i]

        mask_img_path = TRAIN_MASKS_DIR + "/" + name.split(".")[0] + '_mask.gif'
        mask_img = ndimage.imread(mask_img_path, mode='L')
        mask_img[mask_img <= 127.5] = 0
        mask_img[mask_img > 127.5] = 1

        assert run_length_encoding(mask_img) == rle_mask
    print "all test cases passed"


if __name__ == '__main__':
    test_run_length_encoding()
