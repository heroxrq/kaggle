import io
import time

import bson
import numpy as np
import skimage.data
from keras.preprocessing.image import ImageDataGenerator

from config import *
from util import *


def test_generator(bson_file, batch_size):
    product_cnt = 0
    pic_cnt = 0
    num_products = 1768182
    num_pics = 3095080
    pics = []
    prods = []

    with open(bson_file, 'rb') as bf:
        data = bson.decode_file_iter(bf)

        for prod in data:
            product_id = prod['_id']

            for picidx, pic in enumerate(prod['imgs']):
                picture = skimage.data.imread(io.BytesIO(pic['picture']))
                pics.append(picture)
                prods.append(product_id)
                pic_cnt += 1

            # guarantee pics in the same prod are in a same batch
            if 0 <= (pic_cnt % batch_size) < 4 or pic_cnt == num_pics:
                yield np.array(prods), np.array(pics)
                pics = []
                prods = []

            product_cnt += 1
            if product_cnt % 1000 == 0 or product_cnt == num_products:
                print("converted {} products {} images".format(product_cnt, pic_cnt))


def predict_and_make_submission():
    start_time = datetime.datetime.now()

    model = load_model(MODEL_FILE, BEST_WEIGHTS_FILE)

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_RAW_DIR,
        target_size=(INPUT_HEIGHT, INPUT_WIDTH),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical',
        seed=SEED)
    class_indices = train_generator.class_indices()
    dump_to_json_file(class_indices, CLASS_INDICES_FILE)

    index_2_classname = {}
    for class_name in class_indices:
        index_2_classname[class_indices[class_name]] = class_name

    submission_file = SUBMISSION_DIR + "/submission-" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(submission_file, 'w') as outfile:
        outfile.write('_id,category_id\n')

        product_cnt = 0
        pic_cnt = 0
        num_products = 1768182

        for prods, pics in test_generator(TEST_BSON_FILE, PREDICT_BATCH_SIZE):
            probs = model.predict_on_batch(pics)

            start_idx = 0

            for i in xrange(1, len(prods) + 1, 1):
                if i == len(prods) or prods[i] != prods[start_idx]:
                    end_idx = i
                    _id = prods[start_idx]
                    prob = np.average(probs[start_idx: end_idx], axis=0)
                    class_index = np.argmax(prob, axis=1)
                    class_name = index_2_classname[class_index]
                    out_line = str(_id) + ',' + class_name + '\n'
                    outfile.write(out_line)

                    pic_cnt += end_idx - start_idx
                    product_cnt += 1
                    if product_cnt % 1000 == 0 or product_cnt == num_products:
                        print("predicted {} products {} images".format(product_cnt, pic_cnt))

                    start_idx = i

    shell_cmd = "zip %s.zip %s" % (submission_file, submission_file)
    os.system(shell_cmd)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print "predict cost time:", cost_time
