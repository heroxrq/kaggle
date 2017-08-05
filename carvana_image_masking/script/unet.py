import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from keras.optimizers import Adam
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from PIL import Image
from config import *

from subprocess import check_output
print(check_output(["ls", "../dataset"]).decode("utf8"))


# # set the necessary directories
# TRAIN_DIR = "../dataset/train/"
# TRAIN_MASKS_DIR = "../dataset/train_masks/"

all_images = os.listdir(TRAIN_DIR)
train_images, validation_images = train_test_split(all_images, train_size=0.9, test_size=0.1)


# # utility function to convert greyscale images to rgb
# def grey2rgb(img):
#     new_img = []
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             new_img.append(list(img[i][j])*3)
#     new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
#     return new_img


def gen_batch_data(data_dir, mask_dir, images, batch_size, dims):
        """
        Generator which is used to read the data from the directory

        :param data_dir: where the actual resized images are kept
        :param mask_dir: where the actual resized masks are kept
        :param images: the filenames of the images from which batches generated
        :param batch_size: self explanatory
        :param dims: the dimensions of the resized images
        """
        while True:
            idx = np.random.choice(len(images), batch_size)
            imgs = []
            labels = []
            for i in idx:
                # images
                image_name = data_dir + "/" + images[i]
                img = load_img(image_name)
                img_array = img_to_array(img) / 255
                imgs.append(img_array)

                # masks
                mask_name = mask_dir + "/" + images[i].split(".")[0] + '_mask.gif'
                mask = load_img(mask_name)
                mask_array = img_to_array(mask) / 255
                labels.append(mask_array[:, :, 0])
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)


train_gen = gen_batch_data(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, train_images, 16, [RESIZED_HEIGHT, RESIZED_WIDTH])
validation_gen = gen_batch_data(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, validation_images, 16, [RESIZED_HEIGHT, RESIZED_WIDTH])

def down(input_layer, filters, pool=True):
    filters = int(filters)
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D(pool_size=(2, 2))(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D(size=(2, 2))(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    return conv2


# Make a customized UNet implementation.
filters = 64
input_layer = Input(shape=[RESIZED_HEIGHT, RESIZED_WIDTH, 3])
residuals = []

# Down 1, 128X128X3
d1, res1 = down(input_layer, filters)
residuals.append(res1)

filters *= 2

# Down 2, 64X64X64
d2, res2 = down(d1, filters)
residuals.append(res2)

filters *= 2

# Down 3, 32X32X128
d3, res3 = down(d2, filters)
residuals.append(res3)

filters *= 2

# Down 4, 16X16X256
d4, res4 = down(d3, filters)
residuals.append(res4)

filters *= 2

# Down 5, 8X8X512
d5 = down(d4, filters, pool=False)

filters /= 2

# Up 1, 8X8X1024
up1 = up(d5, residual=residuals[-1], filters=filters)

filters /= 2

# Up 2,  16X16X512
up2 = up(up1, residual=residuals[-2], filters=filters)

filters /= 2

# Up 3, 32X32X256
up3 = up(up2, residual=residuals[-3], filters=filters)

filters /= 2

# Up 4, 64X64X128
up4 = up(up3, residual=residuals[-4], filters=filters)

# conv, 128X128X64
out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

model = Model(inputs=input_layer, outputs=out)

model.summary()


# Now let's use Tensorflow to write our own dic
#
#
#
# e_coeficcient metric
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=[dice_coef])
model.fit_generator(train_gen, steps_per_epoch=100, epochs=30)
metric = model.evaluate_generator(validation_gen, steps=100)
print "validation metric", metric

json_string = model.to_json()
model_file = MODEL_DIR + "/model.json"
weights_file = MODEL_DIR + "/weights.h5"
open(model_file, 'w').write(json_string)
model.save_weights(weights_file)

# del model
# model_new = model_from_json(json_string)
# model_new.load_weights(weights_file)


def gen_test_batch_data(data_dir, out_dir, batch_size):
    all_test_images = os.listdir(data_dir)
    for i in range(0, len(all_test_images), batch_size):
        imgs = []
        for j in xrange(batch_size):
            idx = i + j
            if idx >= len(all_test_images):
                break
            image_name = data_dir + "/" + all_test_images[idx]
            img = load_img(image_name)
            img_array = img_to_array(img) / 255
            imgs.append(img_array)
        imgs = np.array(imgs)

        res = model.predict_on_batch(imgs)
        res_imgs = np.reshape(res, (len(res), RESIZED_HEIGHT, RESIZED_WIDTH))

        for k in xrange(len(res)):
            img = Image.fromarray(res_imgs[k] * 255, 'F')
            resized_test_predict_filename = out_dir + "/" + all_test_images[i + k].split(".")[0] + ".gif"
            print i + k, resized_test_predict_filename
            img.save(resized_test_predict_filename)


print "predicting..."
gen_test_batch_data(RESIZED_TEST_DIR, RESIZED_TEST_PREDICT_DIR, 32)
print "end predict..."

########################################################################
original_img = load_img(RESIZED_TEST_DIR + "/" + '913c7bf7de08_14.jpg')
resized_img = imresize(original_img, [128, 128]+[3])
array_img = img_to_array(resized_img)/255
array_img = np.reshape(array_img, (1, 128, 128, 3))
res = model.predict(array_img)
print res
print type(res)
print res.shape

res_img = np.reshape(res, (128, 128))
print type(res_img)
print res_img.shape
print res_img.max()
print res_img.min()
print res_img * 255

img = Image.fromarray(res_img * 255, 'F')
if img.mode != 'RGB':
    img = img.convert('RGB')
img.save('./913c7bf7de08_14.jpg')
img.show()

img = Image.fromarray(res_img * 255, 'F')
img.save('./913c7bf7de08_14.gif')
img.show()
