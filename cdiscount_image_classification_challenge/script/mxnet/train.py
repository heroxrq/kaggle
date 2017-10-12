import argparse
import logging

import data
import fit
from config import *

logging.basicConfig(level=logging.DEBUG)
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '8'


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network             = 'resnext',
        num_layers          = 50,
        # data
        data_train_imglist  = TRAIN_ALL_DIR + "/train_all_train.lst",
        data_train_imgrec   = TRAIN_ALL_DIR + "/train_all_train.rec",
        data_train_imgidx   = TRAIN_ALL_DIR + "/train_all_train.idx",
        data_val_imglist    = TRAIN_ALL_DIR + "/train_all_val.lst",
        data_val_imgrec     = TRAIN_ALL_DIR + "/train_all_val.rec",
        data_val_imgidx     = TRAIN_ALL_DIR + "/train_all_val.idx",
        num_classes         = NUM_CLASSES,
        num_examples        = NUM_TRAIN_IMGS,
        image_shape         = '3,%d,%d' % (INPUT_HEIGHT, INPUT_WIDTH),
        # train
        gpus                = '0,1,2,3',
        batch_size          = TRAIN_BATCH_SIZE,
        num_epochs          = EPOCHS,
        lr                  = LR,
        lr_factor           = 0.5,
        lr_step_epochs      = '5,10,15,20,25,30,35,40,45',
        optimizer           = 'sgd',
        disp_batches        = 10,
        model_prefix        = MODEL_DIR + "/resnext50",
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module(args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_image_iter)
