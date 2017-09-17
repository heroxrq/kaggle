import datetime
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from config import *
from dataset_util import train_data_generator
from loss import dice_coef, weighted_bce_dice_loss
from my_unet import UNet

callbacks = [EarlyStopping(monitor='val_dice_coef',
                           patience=10,
                           verbose=1,
                           min_delta=0.0001,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_dice_coef',
                               factor=0.5,
                               patience=3,
                               verbose=1,
                               epsilon=0.0001,
                               cooldown=0,
                               mode='max'),
             ModelCheckpoint(monitor='val_dice_coef',
                             filepath=BEST_WEIGHTS_FILE,
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max'),
             TensorBoard(log_dir=TF_LOG_DIR)]


def train():
    start_time = datetime.datetime.now()

    all_train_images = os.listdir(INPUT_TRAIN_DIR)
    train_images, validation_images = train_test_split(all_train_images, train_size=0.8, test_size=0.2, random_state=42)

    print "Number of train_images: {}".format(len(train_images))
    print "Number of validation_images: {}".format(len(validation_images))

    train_gen = train_data_generator(INPUT_TRAIN_DIR, INPUT_TRAIN_MASKS_DIR, train_images, TRAIN_BATCH_SIZE, augment=True)
    validation_gen = train_data_generator(INPUT_TRAIN_DIR, INPUT_TRAIN_MASKS_DIR, validation_images, TRAIN_BATCH_SIZE, augment=False)

    model = UNet(layers=LAYERS,
                 input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3),
                 filters=FILTERS,
                 num_classes=1,
                 shrink=True,
                 activation='relu').create_unet_model()
    model.compile(optimizer=RMSprop(lr=0.0005), loss=weighted_bce_dice_loss, metrics=[dice_coef])
    save_model(model, MODEL_FILE)

    steps_per_epoch = len(train_images) / TRAIN_BATCH_SIZE
    validation_steps = len(validation_images) / TRAIN_BATCH_SIZE
    print "steps_per_epoch:", steps_per_epoch
    print "validation_steps:", validation_steps

    model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=validation_gen, validation_steps=validation_steps)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print "train cost time:", cost_time

    return model


def save_model(model, model_file):
    model_json_string = model.to_json()
    with open(model_file, 'w') as mf:
        mf.write(model_json_string)


if __name__ == '__main__':
    train()
