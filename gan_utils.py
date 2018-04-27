#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: gan_utils.py
@time: 2018/4/24 15:06
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Base_Generator(object):
    def __init__(self,iterator,config):
        self.z = iterator.get_next()


if __name__ == '__main__':
    pass