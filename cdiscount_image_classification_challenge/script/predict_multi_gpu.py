import datetime
import io
import time
from Queue import Empty
from multiprocessing import Process, Queue

import bson
import numpy as np

from config import *
from util import load_model, load_img_array, get_class_indices


def test_data_loader(bson_file, batch_size, img_queue):
    product_cnt = 0
    pic_cnt = 0
    num_products = NUM_TEST_PRODUCTS
    num_pics = NUM_TEST_PICS
    prods = []
    pics = []

    with open(bson_file, 'rb') as bf:
        data = bson.decode_file_iter(bf)

        for prod in data:
            product_id = prod['_id']

            for picidx, pic in enumerate(prod['imgs']):
                picture = load_img_array(io.BytesIO(pic['picture']))
                pics.append(picture)
                prods.append(product_id)
                pic_cnt += 1

            # guarantee pics in the same prod are in the same batch
            if 0 <= (pic_cnt % batch_size) < 4 or pic_cnt == num_pics:
                img_queue.put((np.array(prods), np.array(pics)))
                prods = []
                pics = []

            product_cnt += 1
            if product_cnt % 1000 == 0 or product_cnt == num_products:
                print("converted {} products {} images".format(product_cnt, pic_cnt))


def predictor(img_queue, prob_queue):
    model = load_model(MODEL_FILE, BEST_WEIGHTS_FILE)
    pic_cnt = 0
    while True:
        try:
            prods, pics = img_queue.get(timeout=60)
        except Empty:
            print("the img_queue is empty")
            break

        probs = model.predict_on_batch(pics)
        prob_queue.put((prods, probs))

        pic_cnt += len(prods)


def submission_creater(prob_queue):
    class_2_index, index_2_class = get_class_indices(TRAIN_RAW_DIR)
    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('_id,category_id\n')

        product_cnt = 0
        pic_cnt = 0
        num_products = NUM_TEST_PRODUCTS

        while True:
            try:
                prods, probs = prob_queue.get(timeout=60)
            except Empty:
                print("the prob_queue is empty")
                break

            start_idx = 0

            for i in xrange(1, len(prods) + 1, 1):
                if i == len(prods) or prods[i] != prods[start_idx]:
                    end_idx = i
                    _id = prods[start_idx]
                    prob = np.average(probs[start_idx: end_idx], axis=0)
                    class_index = np.argmax(prob, axis=0)
                    class_name = index_2_class[class_index]
                    out_line = str(_id) + ',' + class_name + '\n'
                    outfile.write(out_line)

                    pic_cnt += end_idx - start_idx
                    product_cnt += 1
                    if product_cnt % 1000 == 0 or product_cnt == num_products:
                        print("created {} product submissions of {} images".format(product_cnt, pic_cnt))

                    start_idx = i

    shell_cmd = "zip %s.zip %s" % (submission_file, submission_file)
    os.system(shell_cmd)


def predict_and_make_submission():
    start_time = datetime.datetime.now()

    img_queue = Queue(10)
    test_data_loader_process = Process(target=test_data_loader, args=(TEST_BSON_FILE, TRAIN_BATCH_SIZE, img_queue))
    test_data_loader_process.start()

    prob_queue = Queue(10)
    predictor_process = Process(target=predictor, args=(img_queue, prob_queue))
    predictor_process.start()

    submission_creater_process = Process(target=submission_creater, args=(prob_queue,))
    submission_creater_process.start()

    test_data_loader_process.join()
    predictor_process.join()
    submission_creater_process.join()

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print("predict cost time: {}".format(cost_time))


if __name__ == '__main__':
    predict_and_make_submission()
