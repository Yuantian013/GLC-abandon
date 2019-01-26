import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import gym
import time
import csv
import matplotlib.pyplot as plt
from cartpole_uncertainty import CartPoleEnv_adv as real_env_uncertainty
from cartpole_uncertainty import CartPoleEnv_adv as real_env_clean
import scipy.io as scio
#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS = 2000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = True
# ENV_NAME = 'CartPole-v2'
# env = real_env_uncertainty()
env = real_env_clean()
# env = gym.make(ENV_NAME)
env = env.unwrapped
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,l1,l2,l3,l4,a):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim * 2 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()


        self.ini_l1=tf.constant_initializer(l1)
        self.ini_l2 = tf.constant_initializer(l2)
        self.ini_l3 = tf.constant_initializer(l3)
        self.ini_l4 = tf.constant_initializer(l4)
        self.ini_a = tf.constant_initializer(a)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.LR_D = tf.placeholder(tf.float32, None, 'LR_D')
        self.labda = tf.placeholder(tf.float32, None, 'Lambda')
        self.a = self._build_a(self.S, )  # 这个网络用于及时更新参数
        self.q = self._build_c(self.S, self.a)  # 这个网络是用于及时更新参数


        with tf.variable_scope('Actor',reuse=True):
         self.weight_l1=tf.get_variable('l1/kernel')
        with tf.variable_scope('Actor', reuse=True):
         self.weight_l2=tf.get_variable('l2/kernel')
        with tf.variable_scope('Actor', reuse=True):
         self.weight_l3=tf.get_variable('l3/kernel')
        with tf.variable_scope('Actor', reuse=True):
         self.weight_l4=tf.get_variable('l4/kernel')
        with tf.variable_scope('Actor', reuse=True):
         self.weight_a=tf.get_variable('a/kernel')

        # tf.assign()


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # self.saver.restore(self.sess, "Model/SRDDPG_V3.ckpt")  # 扰动的最好模型
        # self.saver.restore(self.sess, "Model/SRDDPG_V3.ckpt")  # 扰动的最好模型
        # self.saver.restore(self.sess, "Model/SRDDPG_V5.ckpt")  # 无扰动的最好模型

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    #action 选择模块也是actor模块
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable,kernel_initializer=self.ini_l1)#原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable,kernel_initializer=self.ini_l2)  # 原始是30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable,kernel_initializer=self.ini_l3)  # 原始是30
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable,kernel_initializer=self.ini_l4)  # 原始是30
            a = tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable,kernel_initializer=self.ini_a)
            return tf.multiply(a, self.a_bound, name='scaled_a')
    #critic模块
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            net_2 = tf.layers.dense(net_1, 128, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_2, 1, trainable=trainable)  # Q(s,a)

    def show_q(self,s,a,r):
        # print(self.sess.run(self.q, {self.S: s[np.newaxis, :],self.a:a[np.newaxis, :]}),r)
        return self.sess.run(self.q, {self.S: s[np.newaxis, :],self.a:a[np.newaxis, :]}),r

###############################  training  ####################################

load_data = 'test.mat'
data = scio.loadmat(load_data)
weight_a = data['a_COM']
weight_l1=data['l1_COM']
# weight_l2=data['l2_COM']
weight_l2=data['L2']
# weight_l3=data['l3_COM']
weight_l3=data['L3']
weight_l4=data['L4']

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
ddpg = DDPG(a_dim, s_dim, a_bound,weight_l1,weight_l2,weight_l3,weight_l4,weight_a)
print(ddpg.sess.run(ddpg.weight_l3[7,:]))
# ddpg.saver.restore(ddpg.sess, "Model/Group_V1.ckpt")  # 扰动的最好模型
EWMA_p=0.95
EWMA=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
t1 = time.time()

# ddpg.net_0=0
# ddpg.net_1=0
# print(ddpg.sess.run(ddpg.net_0))
# print(ddpg.net_0)
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        # print('start',time.time() - t1)
        a = ddpg.choose_action(s)
        # print('end', time.time() - t1)
        # a = np.clip(np.random.normal(a, 3.5), -a_bound, a_bound)
        s_, r, done, hit = env.step(a)
        # print(r)
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:

            print('Episode:', i, ' Reward: %i' % int(ep_reward))

        elif done:
            if hit==1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), "break in : ", j, "due to ",
                      "hit the wall")
            else:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), "break in : ", j, "due to",
                      "fall down")
            break
