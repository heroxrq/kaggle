import argparse
import logging

import data
import fit
import mxnet as mx
from config import *

logging.basicConfig(level=logging.DEBUG)
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '8'


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)

    parser.add_argument('--pretrained-model', type=str, default='imagenet1k-resnext-101',
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')

    parser.set_defaults(
        # network
        network             = 'resnext',
        num_layers          = 101,
        # data
        data_train_imglist  = TRAIN_ALL_DIR + "/train_all_train.lst",
        data_train_imgrec   = TRAIN_ALL_DIR + "/train_all_train.rec",
        data_train_imgidx   = TRAIN_ALL_DIR + "/train_all_train.idx",
        data_val_imglist    = TRAIN_ALL_DIR + "/train_all_val.lst",
        data_val_imgrec     = TRAIN_ALL_DIR + "/train_all_val.rec",
        data_val_imgidx     = TRAIN_ALL_DIR + "/train_all_val.idx",
        num_classes         = NUM_CLASSES,
        num_examples        = NUM_TRAIN_IMGS,
        image_shape         = '3,128,128',
        # train
        gpus                = '0,1,2,3',
        batch_size          = 400,
        num_epochs          = 10,
        lr                  = 0.01,
        lr_factor           = 0.1,
        lr_step_epochs      = '5',
        wd                  = 0,
        mom                 = 0,
        optimizer           = 'sgd',
        disp_batches        = 10,
        model_prefix        = MODEL_DIR + "/resnext101",
    )

    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)

    args = parser.parse_args()

    # pre-trained model should be downloaded and renamed
    sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL_DIR + "/resnext101", 1)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, args.num_classes, args.layer_before_fullc)

    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_image_iter,
            arg_params  = new_args,
            aux_params  = aux_params)
