import os

from keras import models


def mkdir_if_not_exist(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


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
