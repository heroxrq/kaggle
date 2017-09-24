import datetime
import os
import sys
import time
from Queue import Empty
from multiprocessing import Process, Queue

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


def predictor(gpu, model_dir, pred_queue):
    print "{} load model {}".format(gpu, model_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu[5:]
    import tensorflow as tf

    with tf.device(gpu):
        pic_cnt = 0
        model = load_model(os.path.join(model_dir, 'model.json'),
                           os.path.join(model_dir, 'best_weights.hdf5'))

        all_test_images = sorted(os.listdir(INPUT_TEST_DIR))
        test_gen = test_data_generator(INPUT_TEST_DIR, all_test_images, PREDICT_BATCH_SIZE)

        for batch in test_gen:
            pred = model.predict_on_batch(batch)
            pred_queue.put(pred)
            pic_cnt += len(pred)
            print "{} predicted {} images".format(gpu, pic_cnt)


def ensemble_models(pred_queues, weights):
    norm_coef = len(weights) / sum(weights)
    all_test_images = sorted(os.listdir(INPUT_TEST_DIR))

    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('img,rle_mask\n')

        i = 0
        while True:
            try:
                preds = []
                for pred_queue in pred_queues:
                    preds.append(pred_queue.get(timeout=120))
            except Empty:
                print("the pred_queue is empty, has processed {} images".format(i))
                break

            # weight the prediction according to the lb score of the model
            pred = np.array([weight * preds[i] for i, weight in enumerate(weights)])
            res_array = np.average(pred, axis=0) * norm_coef

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
            i += len(res_array)
    shell_cmd = "zip %s.zip %s" % (submission_file, submission_file)
    os.system(shell_cmd)


def predict_and_make_submission():
    start_time = datetime.datetime.now()

    args = sys.argv[1:]
    argc = len(args)
    assert 2 <= argc <= 8 and argc % 2 == 0, "params is not correct"

    model_dirs = args[:argc/2]
    weights = [float(weight) for weight in args[argc/2:]]

    processes = []
    pred_queues = []

    for i, model_dir in enumerate(model_dirs):
        pred_queue = Queue(10)
        pred_queues.append(pred_queue)
        gpu = "/gpu:{}".format(i)
        predictor_process = Process(target=predictor, args=(gpu, model_dir, pred_queue))
        processes.append(predictor_process)

    ensemble_models_process = Process(target=ensemble_models, args=(pred_queues, weights))
    processes.append(ensemble_models_process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print("predict cost time: {}".format(cost_time))


if __name__ == '__main__':
    predict_and_make_submission()
