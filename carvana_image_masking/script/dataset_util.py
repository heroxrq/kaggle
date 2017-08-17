import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from config import *


def load_img_array(image_name, grayscale=False, target_size=None):
    img = image.load_img(image_name, grayscale, target_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array


def random_hue_saturation_value(image,
                                hue_shift_limit=(-50, 50),
                                sat_shift_limit=(-5, 5),
                                val_shift_limit=(-15, 15),
                                u=0.5):

    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_transform(image,
                     mask,
                     seed,
                     rotation_range=5,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     horizontal_flip=True,
                     u=0.5):
    if np.random.random() < u:
        data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             horizontal_flip=horizontal_flip)
        image_data_gen = ImageDataGenerator(**data_gen_args)
        mask_data_gen = ImageDataGenerator(**data_gen_args)

        image = image_data_gen.random_transform(image, seed)
        mask = mask_data_gen.random_transform(mask, seed)

    return image, mask


def data_augmentation(image, mask, seed):
    image = random_hue_saturation_value(image)
    image, mask = random_transform(image, mask, seed)
    return image, mask


def train_data_generator(data_dir, mask_dir, images, batch_size, target_size=None, augment=False, seed=1):
    batch_cnt = 0
    while True:
        idx = np.random.choice(len(images), batch_size)
        image_batch = []
        mask_batch = []
        for i in idx:
            # images
            image_name = data_dir + "/" + images[i]
            image_array = load_img_array(image_name, grayscale=False, target_size=target_size)

            # masks
            mask_name = mask_dir + "/" + images[i].split(".")[0] + '_mask.gif'
            mask_array = load_img_array(mask_name, grayscale=True, target_size=target_size)

            if augment:
                image_array, mask_array = data_augmentation(image_array, mask_array, seed + batch_cnt)

            image_batch.append(image_array)
            mask_batch.append(mask_array)

            batch_cnt += 1
        image_batch = np.array(image_batch, np.float32)
        mask_batch = np.array(mask_batch, np.float32)
        yield image_batch, mask_batch


def test_data_generator(data_dir, images, batch_size, target_size=None):
    for base in xrange(0, len(images), batch_size):
        image_batch = []
        for offset in xrange(batch_size):
            idx = base + offset
            if idx >= len(images):
                break
            image_name = data_dir + "/" + images[idx]
            image_array = load_img_array(image_name, grayscale=False, target_size=target_size)
            image_batch.append(image_array)
        image_batch = np.array(image_batch, np.float32)
        yield image_batch


def image_resize(in_dir, out_dir, resized_width, resized_height, format=None, save_postfix=None):
    files = os.listdir(in_dir)
    for f in files:
        in_filename = in_dir + os.sep + f
        if save_postfix is None:
            out_filename = out_dir + os.sep + f
        else:
            out_filename = out_dir + os.sep + f.split(".")[0] + save_postfix
        with Image.open(in_filename) as img:
            resized_img = img.resize((resized_width, resized_height))
            resized_img.save(out_filename, format)


def resize_all_images():
    image_resize(TRAIN_DIR, RESIZED_TRAIN_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    # image_resize(TRAIN_MASKS_DIR, RESIZED_TRAIN_MASKS_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)
    image_resize(TEST_DIR, RESIZED_TEST_DIR, RESIZED_WIDTH, RESIZED_HEIGHT)


if __name__ == '__main__':
    resize_all_images()
