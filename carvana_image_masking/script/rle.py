import numpy as np
import pandas as pd
from scipy import ndimage
import os
import tensorflow as tf
from config import *

def run_length_encoding(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)




def test_rle_encode():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
    print rle_to_string(run_length_encoding(test_mask))
    assert rle_to_string(run_length_encoding(test_mask)) == '7 2 11 3'

    train_masks = pd.read_csv(TRAIN_MASKS_CSV)
    for i in xrange(len(train_masks)):
        name = train_masks['img'][i]
        rle_mask = train_masks['rle_mask'][i]
        print rle_mask

        mask_img_path = TRAIN_MASKS_DIR + "/" + name.split(".")[0] + '_mask.gif'
        print mask_img_path
        mask_img = ndimage.imread(mask_img_path, mode='L')
        mask_img[mask_img <= 127] = 0
        mask_img[mask_img > 127] = 1

        print rle_to_string(run_length_encoding(mask_img))
        assert rle_to_string(run_length_encoding(mask_img)) == rle_mask



        return 0






def gen_rle():
    files = os.listdir(TEST_PREDICT_DIR)
    with open(SUBMISSION_DIR + "/" + "submission-2017-08-06", 'w') as outfile:
        outfile.write('img,rle_mask\n')
        i = 0
        for f in files:
            i += 1
            in_filename = TEST_PREDICT_DIR + os.sep + f
            img = ndimage.imread(in_filename, mode='F')

            # print type(img)
            # print img.shape
            # print img.max()
            # print img.min()
            img[img <= 127] = 0
            img[img > 127] = 1
            img = img.astype(int)
            # print type(img)
            # print img.shape
            # print img.max()
            # print img.min()

            rle_str = rle_to_string(run_length_encoding(img))

            # print f
            # print in_filename

            out_string = f.split("_mask")[0] + ".jpg" + ',' + rle_str + '\n'
            outfile.write(out_string)

            if i % 100 == 0:
                print i



def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


if __name__ == '__main__':
    # test_rle_encode()
    # read_train_masks()
    gen_rle()


