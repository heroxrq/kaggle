import sys

from dataset_util import *
from predict import *
from train import *


def main(argv):
    # resize_all_images()  # This is only needed for the first time.
    train()
    predict_and_make_submission()


if __name__ == '__main__':
    main(sys.argv)
