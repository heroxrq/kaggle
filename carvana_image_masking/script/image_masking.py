import numpy as np
import os
import sys
import time
from PIL import Image
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from dataset_util import *
from my_unet import UNet
from rle import *
from config import *
from keras.preprocessing.image import ImageDataGenerator
# ,
# rescale=1./255

# data_gen_args = dict(rotation_range=5.,
#                      width_shift_range=0.05,
#                      height_shift_range=0.05,
#                      zoom_range=0.05)
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)
#
# # Provide the same seed and keyword arguments to the flow methods
# seed = 1
#
# image_generator = image_datagen.flow_from_directory(
#     RESIZED_TRAIN_DIR,
#     target_size=(RESIZED_HEIGHT, RESIZED_WIDTH),
#     color_mode='rgb',
#     class_mode=None,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     save_to_dir=RESIZED_TRAIN_AUGMENTATION_DIR,
#     seed=seed)
#
# mask_generator = mask_datagen.flow_from_directory(
#     RESIZED_TRAIN_MASKS_DIR,
#     target_size=(RESIZED_HEIGHT, RESIZED_WIDTH),
#     color_mode='grayscale',
#     class_mode=None,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     save_to_dir=RESIZED_TRAIN_MASKS_AUGMENTATION_DIR,
#     seed=seed)
#
# print RESIZED_TRAIN_DIR
# print RESIZED_TRAIN_MASKS_DIR
#
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=4,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               cooldown=2,
                               verbose=1),
             ModelCheckpoint(filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir=LOG_DIR)]

def train():
    all_train_images = os.listdir(RESIZED_TRAIN_DIR)
    train_images, validation_images = train_test_split(all_train_images, train_size=0.9, test_size=0.1, random_state=42)

    print "Number of train_images: {}".format(len(train_images))
    print "Number of validation_images: {}".format(len(validation_images))

    train_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, train_images, BATCH_SIZE)
    validation_gen = train_data_generator(RESIZED_TRAIN_DIR, RESIZED_TRAIN_MASKS_DIR, validation_images, BATCH_SIZE)

    model = UNet(layers=6, input_shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 3), filters=32).create_unet_model()
    model.compile(optimizer=Adam(lr=0.001, decay=0.1), loss='binary_crossentropy', metrics=[dice_coef])

    steps_per_epoch = len(train_images) / BATCH_SIZE
    validation_steps = len(validation_images) / BATCH_SIZE
    print "steps_per_epoch:", steps_per_epoch
    print "validation_steps:", validation_steps
    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=validation_gen, validation_steps=validation_steps)

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


def predict_and_make_submission(model):
    all_test_images = os.listdir(RESIZED_TEST_DIR)
    test_gen = test_data_generator(RESIZED_TEST_DIR, all_test_images, BATCH_SIZE)

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
                rle_str = rle_to_string(run_length_encoding(img))

                # make submission
                idx = i + k
                out_string = all_test_images[idx] + ',' + rle_str + '\n'
                outfile.write(out_string)

                if idx % 100 == 0:
                    print idx
            i += BATCH_SIZE


def main(argv):
    # resize_all_images()
    model = train()
    # model = load_model()
    predict_and_make_submission(model)
    # image_resize(RESIZED_TEST_PREDICT_DIR, TEST_PREDICT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
    # gen_rle()


if __name__ == '__main__':
    main(sys.argv)
