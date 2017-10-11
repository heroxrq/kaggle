import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import fit, data
from config import *

import os

os.environ['MXNET_CPU_WORKER_NTHREADS'] = '8'


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'resnext',
        num_layers     = 50,
        kv_store       = 'device',
        test_io        = 0,
        # data
        data_train_dir = TRAIN_RAW_DIR,
        data_train_list= TRAIN_DIR + "/train_raw.lst",
        data_val_dir   = VALID_RAW_DIR,
        data_val_list  = VALID_DIR + "/valid_raw.lst",
        num_classes    = NUM_CLASSES,
        num_examples   = NUM_TRAIN_IMGS,
        image_shape    = '3,96,96',
        # train
        gpus           = '0',
        batch_size     = 800,
        num_epochs     = 50,
        lr             = 0.01,
        lr_factor      = 0.2,
        lr_step_epochs = '10,20,30',
        optimizer      = 'sgd',
        disp_batches   = 10,
        model_prefix   = MODEL_DIR + "/resnext50",
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module(args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_image_iter)
