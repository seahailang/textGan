#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: net_utils.py
@time: 2018/4/20 10:12
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.metrics import f1_score,accuracy_score
import numpy as np

import abc

# network module
class _Base_Module(object):
    def __init__(self, layers,scope=None):
        self.layers = layers
        self.scope = scope
        self._trainable_variables = []

    @abc.abstractmethod
    def call(self, X):
        return X

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            return self.call(X)

    @property
    def trainable_variables(self):
        for i in range(len(self.layers)):
            self._trainable_variables.extend(self.layers[i].trainable_variables)
            return self._trainable_variables


class Flow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X = self.layers[i](X)
        return X


class ResFlow(Flow):
    def call(self,X,additive=False):
        X_ = super().call(X)
        if additive:
            X = X_+X
        else:
            X = tf.concat([X_,X],axis=-1)
        return X


class DenseFlow(_Base_Module):
    def call(self,X):
        for i in range(len(self.layers)):
            X_= self.layers[i](X)
            X = tf.concat([X_,X],axis=-1)
        return X


class Block(_Base_Module):
    def call(self,X):
        X_ = []
        for i in range(len(self.layers)):
            X_.append(self.layers[i](X))
        X = tf.concat(X_,axis=-1)
        return X

# model
class Base_Model(object):
    def __init__(self,iterator,config,**kwargs):

        self.max_train_steps = config.max_train_steps
        self.val_steps = config.val_steps


        self.global_step = tf.train.get_or_create_global_step()
        if config.learning_rate_decay:
            self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                            global_step = self.global_step,
                                                            decay_steps=config.decay_steps,
                                                            decay_rate=config.decay_rate)
        else:
            self.learning_rate = config.learning_rate
        if config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=config.beta1,
                                                    beta2=config.beta2,
                                                    epsilon=config.epsilon)
        elif config.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=config.accumulator_value)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.mode = config.mode
        if self.mode == 'train':
            self.X,self.Y = iterator.get_next()
        else:
            self.X = iterator.get_next()
        self.gpu_config = tf.ConfigProto()
        config.gpu_opyions.allow_growth = True
        self.ckpt = config.ckpt
        self.ckpt_name = config.ckpt_name

    @abc.abstractmethod
    def build_graph(self):
        self.logit = None
        return self.logit

    def loss(self):
        loss = tf.losses.softmax_cross_entropy(self.Y,self.logit)
        return loss

    def compute_gradients(self,loss,var_list=None):
        grads_and_vars = self.optimizer.compute_gradients(loss=loss,var_list=var_list)
        return grads_and_vars

    def apply_gradients(self,grads_and_vars,global_step=None):
        if not global_step:
            global_step = self.global_step
        apply_op = self.optimizer.apply_gradients(grads_and_vars,global_step)
        return apply_op
    def saver(self,var_list=None):
        return tf.train.Saver(var_list=var_list,filename=self.ckpt_name)

    def initializer(self):
        return tf.global_variables_initializer()

    def trainable_variables(self):
        return tf.trainable_variables()


def train_val(model,handle_holder,train_iterator,val_iterator):
    train_summary = []
    val_summary = []
    logit = model.build_graph()
    prob = tf.nn.softmax(logit)
    pred = tf.argmax(prob)
    global_step = model.global_step
    loss = model.loss()
    val_loss_holder = tf.placeholder(dtype=tf.float32,shape=[])
    val_acc_holder = tf.placeholder(tf.float32,[])
    val_summary.append(tf.summary.scalar('val_loss',val_loss_holder))
    val_summary.append(tf.summary.scalar('val_accuracy',val_acc_holder))
    train_summary.append(tf.summary.scalar('train_loss',loss))
    grads_and_vars = model.compute_gradients(loss)
    run_op = model.apply_gradients(grads_and_vars,global_step)
    saver = model.saver()
    initializer = model.initializer()
    train_summary_op = tf.summary.merge(train_summary)
    val_summary_op = tf.summary.merge(val_summary)
    writer = tf.summary.FileWriter(logdir=model.ckpt)
    writer.add_graph(graph = tf.get_default_graph)
    with tf.Session() as sess:
        sess.run(initializer)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        ckpt = tf.train.latest_checkpoint(model.ckpt,model.ckpt_name)
        if ckpt:
            saver.restore(sess,ckpt)
            print('load model from %s'%ckpt)

        for i in range(model.max_train_steps):
            g,l,summary,_ = sess.run([global_step,loss,train_summary_op,run_op],feed_dict={handle_holder:train_handle})

            if g% model.val_steps ==0:
                writer.add_summary(summary, g)
                print('%d,batch_loss:\t %f'%(g,l))
                groud_truth = []
                predictions = []
                val_losses = []
                while True:
                    try:
                        val_label,val_pred,val_loss = sess.run([model.Y,pred,loss],feed_dict={handle_holder:val_handle})
                        groud_truth.extend(list(val_label))
                        predictions.extend(list(val_pred))
                        val_losses.append(val_loss)
                    except:
                        break
                v_loss = np.mean(val_losses)
                v_acc = accuracy_score(groud_truth,predictions)
                summary = sess.run(val_summary_op,feed_dict={val_loss_holder:v_loss,val_acc_holder:v_acc})
                writer.add_summary(summary,g)

                print('validation loss:\t%f'%f1_score(groud_truth,predictions))
                print('validation accuracy:\t%f'%accuracy_score(groud_truth,predictions))
                saver.save(sess,model.ckpt,global_step=g,latest_filename=model.ckpt_name)


def train(model):
    logit = model.build_graph()
    global_step = model.global_step
    loss = model.loss()
    grads_and_vars = model.compute_gradients(loss)
    run_op = model.apply_gradients(grads_and_vars, global_step)
    saver = model.saver()
    initializer = model.initializer()
    with tf.Session() as sess:
        sess.run(initializer)

        ckpt = tf.train.latest_checkpoint(model.ckpt, model.ckpt_name)
        if ckpt:
            saver.restore(sess, ckpt)
            print('load model from %s' % ckpt)

        for i in range(model.max_train_steps):
            g, l, _ = sess.run([global_step, loss, run_op])
            if g % model.val_steps == 0:
                print('%d,batch_loss:\t %f' % (g, l))

def inference(model):
    logit = model.build_graph()
    prob = tf.nn.softmax(logit)
    pred = tf.argmax(prob)
    saver = model.saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(model.ckpt,model.ckpt_name)
        saver.restore(sess,ckpt)
        print('load model from %s' % ckpt)
        results = []
        results_prob = []
        while True:
            try:
                prediction,probability = sess.run([pred,prob])
                results.extend(list(prediction))
                results_prob.extend(list(probability))
            except:
                break
    return results,results_prob











if __name__ == '__main__':
    pass
