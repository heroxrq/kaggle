import datetime
import numpy as np
import os
import sys
import time
from PIL import Image
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from dataset_util import *
from loss import *
from my_unet import UNet
from rle import *

callbacks = [EarlyStopping(monitor='val_dice_coef',
                           patience=4,
                           verbose=1,
                           min_delta=0.0001,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_dice_coef',
                               factor=0.5,
                               patience=2,
                               verbose=1,
                               epsilon=0.0001,
                               mode='max'),
             ModelCheckpoint(monitor='val_dice_coef',
                             filepath=BEST_WEIGHTS_FILE,
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max'),
             TensorBoard(log_dir=LOG_DIR)]


def train():
    start_time = datetime.datetime.now()

    all_train_images = os.listdir(RESIZED_TRAIN_DIR)
    train_images, validation_images = train_test_split(all_train_images, train_size=0.9, test_size=0.1, random_state=42)

    print "Number of train_images: {}".format(len(train_images))
    print "Number of validation_images: {}".format(len(validation_images))

    train_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, train_images, TRAIN_BATCH_SIZE, target_size=(RESIZED_HEIGHT, RESIZED_WIDTH), augment=True)
    validation_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, validation_images, TRAIN_BATCH_SIZE, target_size=(RESIZED_HEIGHT, RESIZED_WIDTH), augment=False)

    model = UNet(layers=LAYERS, input_shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 3), filters=FILTERS, num_classes=1).create_unet_model()
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9), loss=bce_dice_loss, metrics=[dice_coef])

    steps_per_epoch = len(train_images) / TRAIN_BATCH_SIZE
    validation_steps = len(validation_images) / TRAIN_BATCH_SIZE
    print "steps_per_epoch:", steps_per_epoch
    print "validation_steps:", validation_steps

    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=validation_gen, validation_steps=validation_steps)

    save_model(model, MODEL_FILE, WEIGHTS_FILE)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print "train cost time:", cost_time

    return model


def save_model(model, model_file, weights_file):
    model_json_string = model.to_json()
    with open(model_file, 'w') as mf:
        mf.write(model_json_string)
    model.save_weights(weights_file)


def load_model(model_file, weights_file):
    model_json_string = ""
    with open(model_file, 'r') as mf:
        for line in mf:
            model_json_string += line
    model = model_from_json(model_json_string)
    model.load_weights(weights_file)
    return model


def predict_and_make_submission(model):
    start_time = datetime.datetime.now()

    all_test_images = os.listdir(RESIZED_TEST_DIR)
    test_gen = test_data_generator(RESIZED_TEST_DIR, all_test_images, PREDICT_BATCH_SIZE, target_size=(RESIZED_HEIGHT, RESIZED_WIDTH))

    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('img,rle_mask\n')

        i = 0
        for batch in test_gen:
            res_array = model.predict_on_batch(batch)

            res_array = np.reshape(res_array, (len(res_array), RESIZED_HEIGHT, RESIZED_WIDTH))
            for k in xrange(len(res_array)):
                # rle
                img = Image.fromarray(res_array[k] * 255.0, 'F')
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                img = np.array(img)
                img[img <= 127.5] = 0
                img[img > 127.5] = 1
                img = img.astype(int)
                rle_str = run_length_encoding(img)

                # make submission
                idx = i + k
                out_line = all_test_images[idx] + ',' + rle_str + '\n'
                outfile.write(out_line)

                if idx % 1000 == 0:
                    print "processed %d images" % idx
            i += PREDICT_BATCH_SIZE
    shell_cmd = "zip %s.zip %s" % (submission_file, submission_file)
    os.system(shell_cmd)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print "predict cost time:", cost_time


def main(argv):
    # resize_all_images()
    train()
    model = load_model(MODEL_FILE, BEST_WEIGHTS_FILE)
    predict_and_make_submission(model)


if __name__ == '__main__':
    main(sys.argv)
