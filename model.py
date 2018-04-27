#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py.py
@time: 2018/4/24 14:55
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import nest
import net_utils
import numpy as np

class Generator(object):
    def __init__(self,config,matrixs=None,**kwargs):
        ##
        ## one should know in this generator,we generate or encode a batch of sequence with time major
        ## time major means the shape of input or output tensor is: [max_len,batch_size,-1]
        ## -1 is vector_size or 1 according to different stage

        # z is a random stage
        # self.z = iterator.get_next()
        # self.z = tf.cast(self.z,dtype=tf.float32)



        self.token_nums = config.token_nums
        self.vector_size = config.vector_size
        if matrixs:
            self.embedding_matrix = tf.get_variable('embedding_matrix',
                                                shape=[self.token_nums,self.vector_size],
                                                dtype = tf.float32,
                                                initializer = tf.constant_initializer(matrixs))
        else:
            self.embedding_matrix = tf.get_variable('embedding_matrix',
                                                    shape=[self.token_nums, self.vector_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer(stddev=1))
        self.rnn_cell_units = config.rnn_cell_units
        self.batch_size = config.batch_size
        self.sequence_max_len = config.sequence_max_len
        self.learning_rate = config.generator_learning_rate
        self.start_token = tf.convert_to_tensor(config.start_token,dtype=tf.int32)
        self.start_token = tf.tile([self.start_token],[self.batch_size])
        # self.start_token = tf.convert_to_tensor(np.random.randint(10,size=12))
        self.end_token = tf.convert_to_tensor(config.end_token,dtype=tf.int32)
        self.end_token = tf.tile([self.end_token],[self.batch_size])

        self.random_state = tf.random_uniform(shape=[self.batch_size,self.rnn_cell_units],minval=0,maxval=1)
        self.random_token = tf.multinomial(logits=tf.ones((self.batch_size,self.token_nums)),num_samples=1)
        self.random_token = tf.cast(tf.squeeze(self.random_token,-1),tf.int32)
        self.last_tokens = tf.ones((self.batch_size),dtype=tf.int32)*(self.token_nums-1)



        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self._trainable_variables = [self.embedding_matrix]
        self.cell = tf.nn.rnn_cell.GRUCell(self.rnn_cell_units, activation=tf.tanh,kernel_initializer=tf.truncated_normal_initializer())
        self.dense = tf.layers.Dense(self.token_nums,kernel_initializer=tf.truncated_normal_initializer())
        self.layers = [self.cell,self.dense]


    def call(self,X,state):
        X,state = self.cell(X,state)
        X = self.dense(X)
        return X,state

    def embedded(self,sequence):
        return tf.nn.embedding_lookup(self.embedding_matrix,sequence)

    def dynamic_decode(self,initial_state):
        current_time = 0
        # begin with a random state
        # state = self.z
        state = initial_state
        # output_ta contains the output logits
        output_ta = tf.TensorArray(dtype=tf.float32,
                                   clear_after_read=False,
                                   size=self.sequence_max_len,
                                   element_shape=[self.batch_size,self.token_nums])

        # token_ta contains the output tokens
        token_ta = tf.TensorArray(dtype=tf.int32,
                                  size = self.sequence_max_len+1,
                                  clear_after_read=False,
                                  element_shape=[self.batch_size])

        state_ta = tf.TensorArray(dtype=tf.float32,
                                   clear_after_read=False,
                                   size=self.sequence_max_len+1,
                                   element_shape=[self.batch_size,self.rnn_cell_units])

        state_ta.write(0,state)
        # state_ta = state_ta.unstack(tf.random_normal([self.sequence_max_len,self.batch_size,self.rnn_cell_units]))
        # start tokens will be write at position 0
        token_ta = token_ta.write(0,value=self.start_token)

        # finished means if end_token exists in a example
        finished = tf.zeros(shape=[self.batch_size])
        finished = tf.cast(finished,tf.bool)


        def loop_fn(time,output_ta,token_ta,state_ta,finished):
            # at each loop,read a token from token_ta,then get its embedding as input
            token = token_ta.read(time)
            state = state_ta.read(time)

            # token,state = self.act(token,state)
            inp = self.embedded(token)
            # input will be send to a rnn cell and then a dense layer
            # and the output will be write in output_ta
            # generator a token using argmax
            # then the token will be write into token_ta at next position
            # so you know the size of token_ta will be max_length + 1
            output,state = self.call(inp,state)

            # state = tf.random_normal(shape=[self.batch_size,self.rnn_cell_units])
            output_ta = output_ta.write(time,output)
            token = tf.stop_gradient(tf.argmax(output,axis=-1,output_type=tf.int32))

            # if finished,token will be end_token,else new_token will be write
            token = tf.where(finished,self.end_token,token)
            # state = tf.where(finished,self.zero_state,state)

            time = time+1
            # if current token is end token ,then finished will be True
            finished =tf.equal(token,self.end_token)
            token_ta = token_ta.write(time,token)
            state_ta = state_ta.write(time, state)


            return time,output_ta,token_ta,state_ta,finished

        def cond_fn(time,output_ta,token_ta,state_ta,finished):
            cond = tf.logical_not(tf.reduce_all(finished))
            cond = tf.logical_and(cond,tf.less(time,self.sequence_max_len))
            return cond

        _,output_ta,token_ta,state_ta,_= tf.while_loop(cond=cond_fn,
                                               body=loop_fn,
                                               loop_vars=[current_time,output_ta,token_ta,state_ta,finished]
                                               )

        # the start token has been delete,so the length of tokens is sequence_max_len
        outputs = output_ta.stack()
        states = tf.slice(state_ta.stack(),[1,0,0],size=[-1,-1,-1])
        tokens = tf.slice(token_ta.stack(),[1,0],size=[-1,-1])
        # tokens = token_ta.stack()
        return outputs,tokens,states

    def act(self,pre_token,state):
        # given pre_token,current state,
        # a new token will be generator by random
        # a state then change
        # at begin, the start token and the initial_state is a pair.
        inp = tf.nn.embedding_lookup(self.embedding_matrix,pre_token)
        output,state = self.call(inp,state)
        logits = tf.log(tf.nn.softmax(output))

        # here we using a multinomial policy to generator actions
        # one can using a random policy to generator actions
        act = tf.cast(tf.multinomial(logits = logits,num_samples=1),tf.int32)
        # act = tf.argmax(output,axis=-1,output_type=tf.int32)

        act = tf.squeeze(act,axis=-1)
        # I don't know what happened in tf.multinomial function,
        # sometime it'll generator a number out of expectation.
        # so here I add a minimum function to assert the all actions is acceptable
        # ---problem solved now
        # ---tf.multinomial get an unexpected return because a Nan input have been given
        # act =tf.minimum(act,self.last_tokens)
        return act,state


    ### actrually there is no need to give a encode function
    ### for generator generate a sequence totally based on its own state

    # def encode(self,sequence,seq_len):
    #     # encode the sequence to a state sequence
    #     # the start token should in sequence to  coincident with act function
    #     # at zero time step init_state and start_token is a pair
    #     init_state = self.z
    #     zero_state = tf.zeros([self.batch_size,self.rnn_cell_units],dtype=tf.float32)
    #     sequence_ta = tf.TensorArray(dtype=tf.int32,
    #                                  size=self.sequence_max_len,
    #                                  clear_after_read=False)
    #     sequence_ta = sequence_ta.unstack(sequence)
    #     state_ta = tf.TensorArray(dtype=tf.float32,
    #                               size=self.sequence_max_len+1,
    #                               clear_after_read=False,
    #                               element_shape=[self.batch_size,self.rnn_cell_units])
    #     state_ta = state_ta.write(0,init_state)
    #     time = 0
    #
    #
    #     def loop_fn(time,state_ta):
    #         state = state_ta.read(time)
    #         token = sequence_ta.read(time)
    #         inp = tf.nn.embedding_lookup(self.embedding_matrix,token)
    #         _,new_state = self.call(inp,state)
    #         finished = tf.less(time, seq_len)
    #         new_state = tf.where(finished,zero_state,new_state)
    #         time = time + 1
    #         state_ta = state_ta.write(time,new_state)
    #
    #         return time,state_ta
    #     def cond_fn(time,state_ta):
    #         return tf.reduce_all(tf.less(time, seq_len))
    #
    #     _,state_ta= tf.while_loop(cond=cond_fn,
    #                                  body=loop_fn,
    #                                  loop_vars=[time,state_ta])
    #     states = tf.slice(state_ta.stack(),begin=[0,0,0],size=[self.sequence_max_len,-1,-1])
    #     return states

    @property
    def trainable_variables(self):
        for i in range(len(self.layers)):
            self._trainable_variables.extend(self.layers[i].trainable_variables)
        return self._trainable_variables


    def supervised_loss(self,targets,logits,seq_len):
        seq_mask = tf.sequence_mask(seq_len,max_len=self.sequence_max_len)
        targets = tf.boolean_mask(targets,seq_mask)
        logits = tf.boolean_mask(logits,seq_mask)
        losses = tf.losses.softmax_cross_entropy(one_hot_labels=targets,logits=logits)
        return losses

    def reinforced_loss(self,actions,logits,rewards):
        prob = tf.nn.softmax(logits)
        # actions = tf.cast(actions,tf.float32)
        actions =tf.one_hot(actions,self.token_nums,dtype=tf.float32)
        # actions = tf.stop_gradient(actions)
        # log_action_probs = tf.log(tf.reduce_sum(tf.multiply(actions,prob),axis=-1))
        # action_rewards = tf.multiply(log_action_probs,rewards)
        # losses = -tf.reduce_mean(action_rewards)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions,logits=logits)
        loss = tf.multiply(loss,rewards)
        loss = - tf.reduce_mean(loss)
        return loss

    def compute_gradients(self,loss,var_list = None):
        grads_and_vars = self.optimizer.compute_gradients(loss,var_list)
        return grads_and_vars

    def apply_gradients(self,grads_and_vars,global_step=None):
        op  =  self.optimizer.apply_gradients(grads_and_vars,global_step)
        return op


class Discriminator(object):
    def __init__(self,config):
        self.conv1d1 = tf.layers.Conv1D(filters=200,kernel_size=5,padding='same',activation=tf.nn.relu)
        self.conv1d2 = tf.layers.Conv1D(filters=200,kernel_size=3,padding='same',activation=tf.nn.relu)
        self.conv1d3 = tf.layers.Conv1D(filters=200,kernel_size=2,padding='same',activation=tf.nn.relu)
        self.dense = tf.layers.Dense(units=1)
        self.layers = [self.conv1d1,self.conv1d2,self.conv1d3,self.dense]

        self.learning_rate = config.discriminator_learning_rate
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    def call(self,X,time_major = True):
        if time_major:
            X = tf.transpose(X,[1,0,2])
        X1 = self.conv1d1(X)
        X2 = self.conv1d2(X)
        X3 = self.conv1d3(X)

        X = tf.concat((X1,X2,X3),axis=-1)
        # max_pooling over time
        X = tf.reduce_max(X,axis=1)
        X = self.dense(X)
        return X

    @property
    def trainable_variables(self):
        variables = []
        for i in range(len(self.layers)):
            variables.extend(self.layers[i].trainable_variables)
        return variables

    def loss(self,targets,logits):
        loss = tf.losses.log_loss(targets,logits)
        return loss

    def compute_gradients(self,loss,var_list=None):
        return self.optimizer.compute_gradients(loss,var_list=var_list)

    def apply_gradients(self,grads_and_vars,global_step=None):
        return self.optimizer.apply_gradients(grads_and_vars,global_step)






class RollOut(object):
    def __init__(self,policy,config):
        self.policy = policy
        self.batch_size = config.batch_size
        self.sequence_max_len = config.sequence_max_len
        self.end_token = tf.convert_to_tensor(config.end_token, dtype=tf.int32)
        self.end_token = tf.tile([self.end_token], [self.batch_size])
        self.roll_nums = config.roll_nums

    def roll(self,sequences,states):
        outputs = []
        def loop_fn(roll_time,token,token_ta,state,finished):
            # the default policy is a random policy followed by the state-action probability
            token,state = self.policy.act(token,state)
            token_ta = token_ta.write(roll_time,token)
            roll_time = roll_time + 1
            return roll_time,token,token_ta,state,finished


        def cond_fn(roll_time,token,token_ta,state,finished):
            cond = tf.less(roll_time,self.sequence_max_len)
            cond = tf.logical_and(cond,tf.logical_not(tf.reduce_all(finished)))
            return cond
        for j in range(0,self.sequence_max_len):
        # for j in [0]:
            token = sequences[j]
            state = states[j]
            roll_time = j
            token_ta = tf.TensorArray(dtype=tf.int32,size=self.sequence_max_len)

            # token before roll time will be write to token_ta directly
            # token after roll time will be compute by a default policy
            token_ta = token_ta.unstack(tf.slice(sequences,[0,0],[roll_time,-1]))
            finished = tf.cast(tf.zeros([self.batch_size]),tf.bool)
            finished = tf.logical_or(tf.equal(token,self.end_token),finished)


            _,_,token_ta,_,_ = tf.while_loop(cond=cond_fn,
                                             body=loop_fn,
                                             loop_vars=[roll_time,token,token_ta,state,finished])
            tokens = token_ta.stack()
            outputs.append(tokens)
        return outputs


class Model(object):
    def __init__(self,matrixs,config):
        self.gen = Generator(config,matrixs)
        self.disc = Discriminator(config)
        self.roll_out = RollOut(self.gen,config)
        self.sequence_max_len = config.sequence_max_len
        self.batch_size = config.batch_size
        self.ckpt = config.ckpt
        self.max_train_steps = config.max_train_steps




    def train(self,iterator,data_iterator):
        global_step = tf.train.get_or_create_global_step()

        random_state = iterator.get_next()

        real_X,seq_len= data_iterator.get_next()
        # transpose to time major
        real_X = tf.transpose(real_X,[1,0])
        token_logits,tokens,states= self.gen.dynamic_decode(random_state)

        real_X = self.gen.embedded(real_X)
        real_label = tf.ones([self.batch_size,1])
        fake_X = self.gen.embedded(tokens)
        fake_label = tf.zeros([self.batch_size,1])

        # for discriminator
        X = tf.concat((real_X,fake_X),axis=1)
        label = tf.concat((real_label,fake_label),axis=0)
        logits = self.disc.call(X,time_major=True)
        prediction = tf.cast(tf.greater(logits,0),tf.float32)
        real_prediction,fake_prediction = tf.split(prediction,[self.batch_size,self.batch_size])
        acc = tf.equal(prediction,label)
        acc = tf.cast(acc,tf.float32)
        acc = tf.reduce_mean(acc)
        tf.summary.scalar('discriminator_accuracy',acc)
        f_acc = tf.equal(fake_prediction,fake_label)
        f_acc = tf.cast(f_acc, tf.float32)
        f_acc =1 - tf.reduce_mean(f_acc)
        tf.summary.scalar('generator_accuracy',f_acc)
        disc_loss = self.disc.loss(label,tf.sigmoid(logits))
        tf.summary.scalar('discriminator_loss',disc_loss)

        disc_vars = self.disc.trainable_variables
        d_grads_and_vars = self.disc.compute_gradients(disc_loss,disc_vars)
        disc_op = self.disc.apply_gradients(d_grads_and_vars)

        # for generator
        rewards = tf.placeholder(tf.float32,shape=[self.sequence_max_len,self.batch_size])
        gen_loss = self.gen.reinforced_loss(actions=tokens,logits=token_logits,rewards=rewards)
        tf.summary.scalar('gen_loss',gen_loss)

        gen_variable = self.gen.trainable_variables
        g_grads_and_vars = self.gen.compute_gradients(gen_loss,var_list=gen_variable)
        gen_op = self.gen.apply_gradients(g_grads_and_vars)

        rewards_op = []
        roll_actions = self.roll_out.roll(tokens,states)
        for i in range(len(roll_actions)):
            inp = self.gen.embedded(roll_actions[i])
            rewards_op.append(self.disc.call(inp))

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.ckpt)
        writer.add_graph(tf.get_default_graph())
        summary_op = tf.summary.merge_all()
        global_step = tf.assign_add(global_step,1)
        with tf.Session() as sess:
            sess.run(init_op)
            ckpt = tf.train.latest_checkpoint(self.ckpt)
            if ckpt:
                saver.restore(sess,ckpt)
            for i in range(self.max_train_steps):
                r = []
                for j in range(self.roll_out.roll_nums):
                    r.append(np.tanh(sess.run(rewards_op)))
                # r with shape [roll_nums,T,B]
                r = np.array(r)
                # r with shape [T,B]
                r = np.mean(r,axis=0)
                r = np.reshape(r,r.shape[:2])

                g_loss,_ = sess.run([gen_loss,gen_op],feed_dict={rewards:r})

                d_loss,_ = sess.run([disc_loss,disc_op])

                g = sess.run(global_step)
                if g %10== 0:
                    s = sess.run(summary_op,feed_dict={rewards:r})
                    writer.add_summary(s,g)
                    saver.save(sess,self.ckpt,g)
                    print(g,g_loss,d_loss)


        # roll_actions = self.roll_out.roll(fake_X,states)
















FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    pass