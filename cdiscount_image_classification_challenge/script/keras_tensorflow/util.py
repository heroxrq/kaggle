import json
import os

from keras import models
from keras.preprocessing import image
from tensorflow.python.client import device_lib


def mkdir_if_not_exist(dir_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)


def load_img_array(image_path, grayscale=False, target_size=None, rescale=None):
    img = image.load_img(image_path, grayscale, target_size)
    if rescale:
        img_array = rescale * image.img_to_array(img)
    else:
        img_array = image.img_to_array(img)
    return img_array


# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def get_class_indices(directory):
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_2_index = dict(zip(classes, range(len(classes))))
    index_2_class = dict(zip(range(len(classes)), classes))
    return class_2_index, index_2_class


def get_gpus():
    gpus = [device.name.encode('utf8') for device in device_lib.list_local_devices() if device.name[:4] == '/gpu']
    return gpus


def dump_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_from_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_model(model, model_file):
    model_json_string = model.to_json()
    with open(model_file, 'w') as mf:
        mf.write(model_json_string)


def load_model(model_file, weights_file):
    model_json_string = ""
    with open(model_file, 'r') as mf:
        for line in mf:
            model_json_string += line
    model = models.model_from_json(model_json_string)
    model.load_weights(weights_file)
    return model
