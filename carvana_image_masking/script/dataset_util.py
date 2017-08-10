import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def load_img_array(image_name, grayscale=False, target_size=None):
    img = image.load_img(image_name, grayscale, target_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array


def train_data_generator(data_dir, mask_dir, images, batch_size, target_size=None, augment=False, seed=1):
    data_gen_args = dict(rotation_range=5,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True)
    image_data_gen = ImageDataGenerator(**data_gen_args)
    mask_data_gen = ImageDataGenerator(**data_gen_args)

    batch_cnt = 0
    while True:
        idx = np.random.choice(len(images), batch_size)
        imgs = []
        labels = []
        for i in idx:
            # images
            image_name = data_dir + "/" + images[i]
            img_array = load_img_array(image_name, grayscale=False, target_size=target_size)
            if augment:
                img_array = image_data_gen.random_transform(img_array, seed + batch_cnt)
            imgs.append(img_array)

            # masks
            mask_name = mask_dir + "/" + images[i].split(".")[0] + '_mask.gif'
            mask_array = load_img_array(mask_name, grayscale=True, target_size=target_size)
            if augment:
                mask_array = mask_data_gen.random_transform(mask_array, seed + batch_cnt)
            labels.append(mask_array)

            batch_cnt += 1
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels


def test_data_generator(data_dir, images, batch_size, target_size=None):
    for base in xrange(0, len(images), batch_size):
        imgs = []
        for offset in xrange(batch_size):
            idx = base + offset
            if idx >= len(images):
                break
            image_name = data_dir + "/" + images[idx]
            img_array = load_img_array(image_name, grayscale=False, target_size=target_size)
            imgs.append(img_array)
        imgs = np.array(imgs)
        yield imgs
