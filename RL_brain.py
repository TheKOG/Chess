from matplotlib.pyplot import get
import numpy as np
from numpy.core.fromnumeric import shape, var
import pandas as pd
import tensorflow.compat.v1 as tf
import pickle
import os
import random
from keras.utils.np_utils import *
from tensorflow import keras

tf.disable_eager_execution()

np.random.seed(1)
tf.set_random_seed(1)
EDGE=10

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            net_name,
            n_features,
            learning_rate=0.01,
            memory_size=500,
    ):
        self.net_name=net_name
        self.n_features = n_features
        self.lr = learning_rate
        self.memory_size = memory_size
        self.memory_counter = 0

        self.memory = np.zeros((self.memory_size, n_features + 1))
        
        self.x = tf.placeholder("float", [None, EDGE, EDGE], name='x')
        self.y = tf.placeholder("float", [None, 1], name='y')

        self.Build_Net()
        self.Build_Train()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.load()
        self.save()

    def Build_Net(self):
        conv_w1 = tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=0.1), dtype=tf.float32, name='wc1')
        conv_w2 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.1), dtype=tf.float32, name='wc2')
        conv_w3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='wc3')
        dense_w1 = tf.Variable(tf.random_normal([256, 128], stddev=0.1), dtype=tf.float32, name='wd1')
        dense_w2 = tf.Variable(tf.random_normal([128, 1], stddev=0.1), dtype=tf.float32, name='wd2')
        conv_b1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='bc1')
        conv_b2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bc2')
        conv_b3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='bc3')
        dense_b1 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bd1')
        dense_b2 = tf.Variable(tf.random_normal([1], stddev=0.1), dtype=tf.float32, name='bd2')
        weights = {
            'wc1': conv_w1,
            'wc2': conv_w2,
            'wc3': conv_w3,
            'wd1': dense_w1,
            'wd2': dense_w2
        }
        biases = {
            'bc1': conv_b1,
            'bc2': conv_b2,
            'bc3': conv_b3,
            'bd1': dense_b1,
            'bd2': dense_b2
        }
        self.q_value = self.Calc(self.x, weights, biases)
    
    
    def Calc(self, _input, _w, _b):
        pps = tf.reshape(_input, shape=[-1, EDGE, EDGE, 1])
        pps = tf.nn.conv2d(pps, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        pps = tf.nn.relu(tf.nn.bias_add(pps, _b['bc1']))
        pps = tf.nn.max_pool(pps, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pps = tf.nn.conv2d(pps, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        pps = tf.nn.relu(tf.nn.bias_add(pps, _b['bc2']))
        pps = tf.nn.max_pool(pps, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pps = tf.nn.conv2d(pps, _w['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        pps = tf.nn.relu(tf.nn.bias_add(pps, _b['bc3']))
        pps = tf.reduce_mean(pps, [1, 2])
        pps = tf.nn.relu(tf.add(tf.matmul(pps, _w['wd1']), _b['bd1']))
        pps = tf.add(tf.matmul(pps, _w['wd2']), _b['bd2'])
        return pps

    def Build_Train(self):
        self.cost = tf.reduce_mean(tf.squared_difference(self.q_value, self.y))
        self.optm = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam').minimize(self.cost)

    def Store_Transition(self, s, r):
        transition = np.hstack((s, [r]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def Forget(self):
        self.memory_counter =0

    def Fall_Flat(self,obs):
        re=np.array(obs).reshape((EDGE,EDGE))
        return re

    def choose_action(self,_observation):
        observation=self.Fall_Flat(_observation)
        boards=[]
        action=[]
        v=[]
        for i in range(EDGE):
            for j in range(EDGE):
                if observation[i][j]!=0:
                    continue
                obs_next=np.copy(observation)
                obs_next[i,j]=1
                boards.append(obs_next)
                action.append(i*EDGE+j)
                v.append(0)
        size=len(boards)
        action_value = self.sess.run(self.q_value, feed_dict={self.x: np.array(boards)})
        max_i=0
        for i in range(size):
            v[i]=action_value[i,0]+random.randint(0, 10)/1000
            if(v[i]>v[max_i]):
                max_i=i
        return action[max_i],v[max_i],boards[max_i]

    def Turn2D(self,data):
        #print(data)
        dshape=data.shape
        re=np.zeros((dshape[0],EDGE,EDGE))
        for i in range(dshape[0]):
            re[i]=self.Fall_Flat(data[i])
        return re

    def learn(self):
        batch_index = np.arange(self.memory_counter, dtype=np.int32)
        batch_memory = self.memory[batch_index,:]
        q_target=np.zeros((self.memory_counter,1))
        q_target[batch_index,0] = batch_memory[batch_index,self.n_features]
        _, cost = self.sess.run([self.optm, self.cost], feed_dict={self.x: self.Turn2D(batch_memory[:, :self.n_features]),self.y: q_target})
        self.cost_his.append(cost)
        self.Forget()


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save(self):
        try:
            self.saver.save(self.sess,self.net_name+"\\brain")
        except:
            pass

    def load(self):
        if os.path.exists(self.net_name+"\\brain.index")==False:
            return
        self.saver.restore(self.sess, os.path.abspath(self.net_name+"\\brain"))