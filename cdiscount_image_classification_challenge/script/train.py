import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator

from config import *
from util import save_model

# --------------------------------------------------
# gpu config
# --------------------------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# --------------------------------------------------
# prepare data
# --------------------------------------------------
train_datagen = ImageDataGenerator()

validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_RAW_DIR,
    target_size=(INPUT_HEIGHT, INPUT_WIDTH),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    seed=SEED)

validation_generator = validation_datagen.flow_from_directory(
    directory=VALID_RAW_DIR,
    target_size=(INPUT_HEIGHT, INPUT_WIDTH),
    batch_size=VALID_BATCH_SIZE,
    class_mode='categorical',
    seed=SEED)

print("train and valid generator is ok")

# --------------------------------------------------
# create and train model
# --------------------------------------------------
callbacks = [EarlyStopping(monitor='val_acc',
                           patience=10,
                           verbose=1,
                           min_delta=0.00001,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_acc',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.00001,
                               cooldown=0,
                               mode='max'),
             ModelCheckpoint(monitor='val_acc',
                             filepath=BEST_WEIGHTS_FILE,
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1,
                             mode='max'),
             TensorBoard(log_dir=TF_LOG_DIR)]

steps_per_epoch = NUM_TRAIN_IMGS / TRAIN_BATCH_SIZE
validation_steps = NUM_VALID_IMGS / VALID_BATCH_SIZE
print("steps_per_epoch: {}".format(steps_per_epoch))
print("validation_steps: {}".format(validation_steps))

# create the base pre-trained model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
# x = Dense(FC_SIZE, activation='relu')(x)
# add a logistic layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(lr=LR1), loss='categorical_crossentropy', metrics=['accuracy'])
save_model(model, MODEL_FILE)

# train the model on the new data for a few epochs
model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS1,
                    callbacks=callbacks,
                    validation_data=validation_generator, validation_steps=validation_steps)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=LR2, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS2,
                    callbacks=callbacks,
                    validation_data=validation_generator, validation_steps=validation_steps)
save_model(model, MODEL_FILE)
