import numpy as np
import os
import sys
from PIL import Image
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from dataset_util import *
from image_util import *
from my_unet import UNet
from rle import *


def train():
    all_train_images = os.listdir(RESIZED_TRAIN_DIR)
    train_images, validation_images = train_test_split(all_train_images, train_size=0.9, test_size=0.1, random_state=42)

    train_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, train_images, BATCH_SIZE)
    validation_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, validation_images, BATCH_SIZE)

    model = UNet(layers=6, input_shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 3), filters=32).create_unet_model()
    model.compile(optimizer=Adam(lr=0.001, decay=0.1), loss='binary_crossentropy', metrics=[dice_coef])

    steps_per_epoch = len(train_images) / BATCH_SIZE
    validation_steps = len(validation_images) / BATCH_SIZE
    print "steps_per_epoch: ", steps_per_epoch
    print "validation_steps: ", validation_steps
    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=3, validation_data=validation_gen, validation_steps=validation_steps)

    model.save_weights(WEIGHTS_FILE)
    model_json_string = model.to_json()
    with open(MODEL_FILE, 'w') as model_file:
        model_file.write(model_json_string)

    return model


def load_model():
    model_json_string = ""
    with open(MODEL_FILE, 'r') as model_file:
        for line in model_file:
            model_json_string += line
    model = model_from_json(model_json_string)
    model.load_weights(WEIGHTS_FILE)
    return model


def predict(model):
    all_test_images = os.listdir(RESIZED_TEST_DIR)
    test_gen = test_data_generator(RESIZED_TEST_DIR, all_test_images, BATCH_SIZE)

    i = 0
    for batch in test_gen:
        res_array = model.predict_on_batch(batch)

        res_array = np.reshape(res_array, (len(res_array), RESIZED_HEIGHT, RESIZED_WIDTH))
        for k in xrange(len(res_array)):
            idx = i + k
            img = Image.fromarray(res_array[k] * 255.0, 'F')
            resized_test_predict_filename = RESIZED_TEST_PREDICT_DIR + "/" + all_test_images[idx].split(".")[0] + "_mask.gif"
            img.save(resized_test_predict_filename)
            print idx, resized_test_predict_filename
        i += BATCH_SIZE


def main(argv):
    # resize_all_images()
    model = train()
    # model = load_model()
    predict(model)
    image_resize(RESIZED_TEST_PREDICT_DIR, TEST_PREDICT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
    gen_rle()


if __name__ == '__main__':
    main(sys.argv)
