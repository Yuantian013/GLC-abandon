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
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim * 2 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

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

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/SRDDPG_V3.ckpt")  # 扰动的最好模型
        # self.saver.restore(self.sess, "Model/Group_V1.ckpt")  # 扰动的最好模型
        # self.saver.restore(self.sess, "Model/SRDDPG_V5.ckpt")  # 无扰动的最好模型

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    #action 选择模块也是actor模块
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            a = tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
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


s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)
EWMA_p=0.95
EWMA=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
t1 = time.time()
Q=np.zeros(MAX_EP_STEPS)
R=np.zeros(MAX_EP_STEPS)
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    T=0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        ts=time.time()
        a = ddpg.choose_action(s)
        te=time.time()
        # a = np.clip(np.random.normal(a, 3.5), -a_bound, a_bound)
        s_, r, done, hit = env.step(a)
        # print(r)
        Q[j],R[j]=ddpg.show_q(s,a,r)
        s = s_
        ep_reward += r
        T += (te-ts)
        if j == MAX_EP_STEPS - 1:

            print('Episode:', i, ' Reward: %i' % int(ep_reward),'t:',T/(j+1))

        elif done:
            if hit==1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), "break in : ", j, "due to ",
                      "hit the wall",'t:',T/(j+1))
            else:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), "break in : ", j, "due to",
                      "fall down",'t:',T/(j+1))
            break
    print("Saved")
    scio.savemat('QR',
                              {'Q': Q,
                               'R': R,})