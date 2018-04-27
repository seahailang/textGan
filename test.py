#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: test.py
@time: 2018/4/25 14:06
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import model
import config
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def sorted_number_gen(max_nums,max_len):
    for i in range(10000000):
        seq_len = np.random.randint(5,max_len)
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


def test_generator():
    generator_config = config.Config()
    data = np.random.randint(10,100,size=(10000,generator_config.rnn_cell_units))

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(generator_config.batch_size)
    iterator = dataset.make_one_shot_iterator()
    gen = model.Generator(iterator,generator_config,matrixs=None)
    output,token,states = gen.dynamic_decode()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(token))
        print(sess.run(states))


def test_mu():
    cell = tf.nn.rnn_cell.GRUCell(100)
    state = tf.random_uniform(shape=(20,100))
    inp = tf.random_uniform(shape=(20,100))
    sess = tf.Session()
    inp, state = cell(inp, state)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100):
        inp,state = cell(inp,state)
        a = tf.multinomial(inp,1)
        print(sess.run(a).max())

def test_roll():
    generator_config = config.Config()
    data = np.random.uniform(10, 100, size=(10000, generator_config.rnn_cell_units))
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(generator_config.batch_size)
    iterator = dataset.make_one_shot_iterator()
    gen = model.Generator(iterator, generator_config, matrixs=None)
    # token = gen.random_token
    # state = gen.random_state
    #
    # act,state = gen.act(token,state)

    output, token, states = gen.dynamic_decode()


    roll = model.RollOut(gen,generator_config)

    output = roll.roll(token,states)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # for i in range(1000):
        #     print(sess.run(act))
    #     # t = sess.run(token)
    #     # print(np.max(t))
    #     print(sess.run(token))
        o = sess.run(output)
        print(o[-1])


def test_model():
    test_config = config.Config()
    data = make_sorted_dataset(test_config.token_nums,test_config.sequence_max_len,test_config.batch_size)
    data_iterator = data.make_one_shot_iterator()

    data = np.random.uniform(-1, 1, size=(10000, test_config.rnn_cell_units))
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(test_config.batch_size)
    iterator = dataset.make_one_shot_iterator()


    test = model.Model(None,test_config)
    test.train(iterator,data_iterator)




if __name__ == '__main__':
    # test_generator()
    tf.set_random_seed(34)
    test_model()
    # test_mu()
