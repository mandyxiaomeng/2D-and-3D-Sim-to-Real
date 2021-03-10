# coding:utf-8

import os
import os.path as osp
from train_source import BATCH_SIZE, num_class


class Config:
    # -------------------- data config --------------------#

    num_classes = num_class
    classes = []
    totality = 0
    num_point = 0

    save_path = ''

    # -------------------- model config --------------------#

    learning_rate = 1e-3
    batch_size = BATCH_SIZE
    decay_rate = 0.7
    decay_step = 200000
    end_point = {}


config = Config()

if __name__ == '__main__':
    print(config.test_files)
