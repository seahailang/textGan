#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: test_dataset.py.py
@time: 2018/4/26 15:18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS


def sorted_number_gen(max_nums,max_len):
    for i in range(10000000):
        seq_len = np.random.randint(80,max_len)
        a = np.random.randint(2,max_nums,size=seq_len)
        a = sorted(a)
        a = a +[0]*(max_len-seq_len)
        # a = np.array(a)
        yield a,seq_len
def make_sorted_dataset(max_nums,max_len,batch_size):
    gen = lambda :sorted_number_gen(max_nums,max_len)
    dataset = tf.data.Dataset.from_generator(gen,output_types=(tf.int32,tf.int32),
                                             output_shapes=(tf.TensorShape([max_len]),tf.TensorShape([])))
    dataset = dataset.repeat().batch(batch_size)
    return dataset

if __name__ == '__main__':
    datasets = make_sorted_dataset(1000,120,2)
    iterator = datasets.make_one_shot_iterator()
    a,b = iterator.get_next()
    sess =tf.Session()
    print(sess.run(a))
    # a= sorted_number_gen(100,120)
    # print(next(a))