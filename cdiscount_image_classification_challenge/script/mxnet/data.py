import mxnet as mx


def get_image_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])

    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)

    train = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        label_width         = 1,
        path_imglist        = args.data_train_list,
        path_root           = args.data_train_dir,
        # shuffle             = True,
        data_name           = 'data',
        label_name          = 'softmax_label',
        num_parts           = nworker,
        part_index          = rank,
        # rand_crop           = True,
        # rand_mirror         = True,
        # mean                = True,
        # brightness          = 0.1,
        # contrast            = 0.1,
        # saturation          = 0.1,
        # hue                 = 0.1,
        # rand_gray           = 0.05,
        # pca_noise           = 0.05
    )

    val = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        label_width         = 1,
        path_imglist        = args.data_val_list,
        path_root           = args.data_val_dir,
        data_name           = 'data',
        label_name          = 'softmax_label',
        num_parts           = nworker,
        part_index          = rank)

    return (train, val)
