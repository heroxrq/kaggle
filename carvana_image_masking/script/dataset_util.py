import numpy as np
from keras.preprocessing import image


def load_img_array(image_name, grayscale):
    img = image.load_img(image_name, grayscale)
    img_array = image.img_to_array(img) / 255.0
    return img_array


def train_data_generator(data_dir, mask_dir, images, batch_size):
    while True:
        idx = np.random.choice(len(images), batch_size)
        imgs = []
        labels = []
        for i in idx:
            # images
            image_name = data_dir + "/" + images[i]
            img_array = load_img_array(image_name, grayscale=False)
            imgs.append(img_array)

            # masks
            mask_name = mask_dir + "/" + images[i].split(".")[0] + '_mask.gif'
            mask_array = load_img_array(mask_name, grayscale=True)
            labels.append(mask_array)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels


def test_data_generator(data_dir, images, batch_size):
    for base in xrange(0, len(images), batch_size):
        imgs = []
        for offset in xrange(batch_size):
            idx = base + offset
            if idx >= len(images):
                break
            image_name = data_dir + "/" + images[idx]
            img_array = load_img_array(image_name, grayscale=False)
            imgs.append(img_array)
        imgs = np.array(imgs)
        yield imgs
