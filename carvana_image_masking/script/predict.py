import datetime
import time
import os
import numpy as np
from keras import models

from config import *
from dataset_util import test_data_generator
from rle import run_length_encoding


def load_model(model_file, weights_file):
    model_json_string = ""
    with open(model_file, 'r') as mf:
        for line in mf:
            model_json_string += line
    model = models.model_from_json(model_json_string)
    model.load_weights(weights_file)
    return model


def predict_and_make_submission():
    start_time = datetime.datetime.now()

    model = models.load_model(BEST_MODEL_FILE)
    all_test_images = os.listdir(INPUT_TEST_DIR)
    test_gen = test_data_generator(INPUT_TEST_DIR, all_test_images, PREDICT_BATCH_SIZE)

    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('img,rle_mask\n')

        i = 0
        for batch in test_gen:
            res_array = model.predict_on_batch(batch)

            res_array = np.reshape(res_array, (len(res_array), OUTPUT_HEIGHT, OUTPUT_WIDTH))
            for k in xrange(len(res_array)):
                # rle
                img = res_array[k]
                img = np.where(img > 0.5, 1, 0)
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


if __name__ == '__main__':
    predict_and_make_submission()
