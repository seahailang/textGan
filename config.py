#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: config.py.py
@time: 2018/4/24 14:55
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Config(object):
    def __init__(self):
        self.token_nums = 1000
        self.vector_size = 200
        self.sequence_max_len = 14
        self.batch_size = 12
        self.generator_learning_rate = 0.01
        self.start_token = 1
        self.end_token = 0
        self.rnn_cell_units = 300
        self.roll_nums = 5
        self.discriminator_learning_rate=0.001
        self.max_train_steps = 100000
        self.ckpt = './ckpt/'


if __name__ == '__main__':
    pass