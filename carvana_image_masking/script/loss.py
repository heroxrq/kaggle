import tensorflow as tf
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred):
    smooth = 1e-3

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))
