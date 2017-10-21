import datetime
import io
import logging
import math
import os
import time
from multiprocessing import Process, Queue

import bson
import mxnet as mx
import numpy as np
from config import *
from mxnet.io import DataBatch, DataIter

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t%(levelname)s\t%(filename)s:%(lineno)d\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mxnet_predict")


def mkdir_if_not_exist(dir_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)


def get_class_index_name(path_imglist):
    index_2_name = dict()
    with open(path_imglist) as f:
        for line in f:
            fields = line.split('\t')
            class_index = int(float(fields[1]))
            class_name = fields[2].split(os.path.sep)[0]
            if class_index not in index_2_name:
                index_2_name[class_index] = class_name
    return index_2_name


def get_prods(bson_file):
    prod_cnt = 0
    pic_cnt = 0
    prods = []

    with open(bson_file, 'rb') as bf:
        data = bson.decode_file_iter(bf)

        for prod in data:
            product_id = prod['_id']

            for picidx, pic in enumerate(prod['imgs']):
                prods.append(product_id)
                pic_cnt += 1

            prod_cnt += 1
    print("totally {} products {} images".format(prod_cnt, pic_cnt))
    return prods


def gen_bson_flatten_file(bson_file, bson_flatten_file):
    with open(bson_file, 'rb') as bf, open(bson_flatten_file, 'wb') as bff:
        data = bson.decode_file_iter(bf)
        for prod in data:
            product_id = prod['_id']

            for picidx, pic in enumerate(prod['imgs']):
                picture = pic['picture']
                pic_name = "{}.{}".format(product_id, picidx)
                bff.write(bson.BSON.encode({'name': pic_name, 'pic': picture}))
    print("{} generated".format(bson_flatten_file))


class TestDataIter(DataIter):
    def __init__(self, bson_flatten_file, data_shape, dtype=np.float32):
        super(TestDataIter, self).__init__()
        self.bson_flatten_file = bson_flatten_file
        self.data_shape = data_shape
        self.dtype = dtype
        self.batch_size = data_shape[0]
        self.bff_iter = bson.decode_file_iter(open(bson_flatten_file, 'rb'))
        self.pic_cnt = 0

    def next(self):
        batch_data = mx.ndarray.empty(self.data_shape)
        i = 0
        if self.pic_cnt < NUM_TEST_PICS:
            for prod in self.bff_iter:
                str_image = io.BytesIO(prod['pic']).read()
                picture = mx.img.resize_short(mx.img.imdecode(str_image), self.data_shape[-1])
                picture = mx.ndarray.transpose(picture, axes=(2, 0, 1))
                batch_data[i] = picture
                i += 1
                self.pic_cnt += 1

                if self.pic_cnt % 1000 == 0 or self.pic_cnt == NUM_TEST_PICS:
                    logger.info("converted {} images".format(self.pic_cnt))

                if self.pic_cnt % self.batch_size == 0:
                    return DataBatch(data=[batch_data],
                                     pad=0,
                                     index=None,
                                     provide_data=self.provide_data)
                elif self.pic_cnt == NUM_TEST_PICS:
                    return DataBatch(data=[batch_data],
                                     pad=self.batch_size - i,
                                     index=None,
                                     provide_data=self.provide_data)
        else:
            raise StopIteration

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data_shape, self.dtype, layout='NCHW')]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.bff_iter = bson.decode_file_iter(open(self.bson_flatten_file, 'rb'))
        self.pic_cnt = 0


def predictor(prob_queue=None):
    batch_size = 100
    data_shape = (batch_size, 3, 128, 128)

    sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL_DIR + "/resnext101", 1)
    mod = mx.mod.Module(
        symbol        = sym,
        context       = [mx.gpu(int(i)) for i in ['0']]
    )
    mod.bind(data_shapes=[('data', data_shape)], for_training=False)
    mod.set_params(arg_params, aux_params)

    bson_flatten_file = os.path.join(DATASET_DIR, 'test_flatten.bson')
    if not os.path.exists(bson_flatten_file):
        gen_bson_flatten_file(TEST_BSON_FILE, bson_flatten_file)

    test_iter = TestDataIter(bson_flatten_file, data_shape)

    num_batch = int(math.ceil(NUM_TEST_PICS / batch_size))
    for pred, i_batch, batch in mod.iter_predict(eval_data=test_iter, reset=False):
        if prob_queue is not None:
            prob_queue.put(pred[0].asnumpy())
            logger.info("predicted {} batches".format(i_batch + 1))


def submission_creater(prob_queue):
    index_2_name = get_class_index_name(TRAIN_ALL_DIR + "/train_all_train.lst")
    prods = get_prods(TEST_BSON_FILE)
    probs = []

    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('_id,category_id\n')

        product_cnt = 0
        pic_cnt = 0
        num_products = NUM_TEST_PRODUCTS

        prob_batch = prob_queue.get(timeout=120)
        for j in range(len(prob_batch)):
            probs.append(prob_batch[j])

        start_idx = 0

        for i in range(1, len(prods) + 1, 1):
            if prods[i] != prods[start_idx] or i == len(prods):
                end_idx = i
                num_pics = end_idx - start_idx
                _id = prods[start_idx]

                prob = np.average(probs[: num_pics], axis=0)
                class_index = np.argmax(prob, axis=0)

                class_name = index_2_name[class_index]
                out_line = str(_id) + ',' + class_name + '\n'
                outfile.write(out_line)

                pic_cnt += num_pics
                product_cnt += 1
                if product_cnt % 1000 == 0 or product_cnt == num_products:
                    print("created {} product submissions of {} images".format(product_cnt, pic_cnt))

                start_idx = i

                for _ in range(num_pics):
                    probs.pop(0)

                if len(probs) < 5:
                    try:
                        prob_batch = prob_queue.get(timeout=30)
                        for j in range(len(prob_batch)):
                            probs.append(prob_batch[j])
                    except Exception:
                        print("the prob_queue is empty")

    shell_cmd = "zip {}.zip {}".format(submission_file, submission_file)
    os.system(shell_cmd)


def predict_and_make_submission():
    start_time = datetime.datetime.now()

    processes = []

    prob_queue = Queue(50)
    predictor_process = Process(target=predictor, args=(prob_queue,))
    processes.append(predictor_process)
    submission_creater_process = Process(target=submission_creater, args=(prob_queue,))
    processes.append(submission_creater_process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print("predict cost time: {}".format(cost_time))


if __name__ == '__main__':
    predict_and_make_submission()
