import tensorflow as tf
import numpy as np
import time
from cartpole_uncertainty import CartPoleEnv_adv as dreamer
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math
#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS =500
LR_D = 0.001
LR_R = 0.001
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False
env = dreamer()

env = env.unwrapped
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        tf.reset_default_graph()
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_D= tf.placeholder(tf.float32, None, 'LR_D')

        self.a = self._build_a(self.S,)# 这个网络用于及时更新参数
        self.q = self._build_c(self.S, self.a, )# 这个网络是用于及时更新参数


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, "Model/SRDDPG_V3.ckpt")  # 1 0.1 0.5 0.001

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self,LR_D):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.dreamertrain, {self.S: bs,self.a: ba, self.S_: bs_, self.LR_D: LR_D})
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

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

class Dreamer(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        tf.reset_default_graph()
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_D= tf.placeholder(tf.float32, None, 'LR_D')
        self.LR_R = tf.placeholder(tf.float32, None, 'LR_R')

        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.dreamer = self._build_dreamer(self.S, self.A, )
        self.score=self._build_score(self.dreamer)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dreamer')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Score')



        self.dreamer_loss_s = tf.reduce_mean(tf.squared_difference(self.S_, self.dreamer))

        self.dreamer_loss_r = tf.reduce_mean(tf.squared_difference(self.R, self.score))

        self.dreamertrain_s = tf.train.AdamOptimizer(self.LR_D).minimize(self.dreamer_loss_s,var_list = d_params)

        self.dreamertrain_r = tf.train.AdamOptimizer(self.LR_R).minimize(self.dreamer_loss_r, var_list=r_params)

        # self.sess.run(tf.reset_default_graph())
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/SRDDPG_D_V1.ckpt")  # 1 0.1 0.5 0.001
    def dream(self, s,a):
        return self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0],self.sess.run(self.score, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0]

    def learn(self,LR_D,LR_R):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        self.sess.run(self.dreamertrain_s, {self.S: bs,self.A: ba, self.S_: bs_, self.LR_D: LR_D})
        self.sess.run(self.dreamertrain_r, {self.S: bs, self.A: ba, self.R: br, self.LR_R: LR_R})

        return self.sess.run(self.dreamer_loss_s, {self.S: bs,self.A: ba, self.S_: bs_}),self.sess.run(self.dreamer_loss_r, {self.S: bs,self.A: ba, self.R: br})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_dreamer(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Dreamer', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_2, self.s_dim, trainable=trainable)

    def _build_score(self, s,reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Score', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512  # 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s)+ b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_2, 1, trainable=trainable)

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/SRDDPG_D_V2.ckpt")
        print("Save to path: ", save_path)
###############################  training  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)
dreamer=Dreamer(a_dim, s_dim, a_bound)
# dreamer.saver.restore(dreamer.sess, "Model/SRDDPG_D.ckpt")  # 1 0.1 0.5 0.001
var = 5  # control exploration
t1 = time.time()
plot=False
min_loss_s=0.001
min_loss_r=0.01
loss_r=100
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    X_=[]
    Theta_ = []
    X_PRE=[]
    Theta__PRE = []
    Reward=[]
    R_PRE=[]
    step=[]
    R_DREAM=[]
    if int(i % 3) == 0:
        ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_V3.ckpt")  # 1 0.1 0.5 0.001
    else:
        ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_BAD.ckpt")  # 1 0.1 0.5 0.001

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
        s_, r, done, hit = env.step(a,i)
        s_pre,r_pre=dreamer.dream(s,a)
        dreamer.store_transition(s, a, r/100, s_)

        x, _, theta, _=s_pre
        r_1 = ((1 - abs(x)))  # - 0.8
        r_2 = (((20 * 2 * math.pi / 360) / 4) - abs(theta)) / ((20 * 2 * math.pi / 360) / 4)  # - 0.5
        reward = np.sign(r_2) * ((10 * r_2) ** 2) + np.sign(r_1) * ((10 * r_1) ** 2)
        # print(s_,s_pre,reward,r,r_pre*100)


        if done:
            break

        if dreamer.pointer > MEMORY_CAPACITY:
            # ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_V3.ckpt")  # 1 0.1 0.5 0.001
            var *= .999995    # decay the action randomness
            LR_D*= .99995
            LR_R *=.99995
            loss_s,loss_r=dreamer.learn(LR_D,LR_R)
            if loss_s <min_loss_s:
                LR_D *= (loss_s/min_loss_s)
                min_loss_s=loss_s
            if loss_r <min_loss_r:
                LR_R *= (loss_r/min_loss_r)
                min_loss_r=loss_r
                dreamer.save_result()
        s = s_
        step.append(j)
        Reward.append(r/100)
        R_PRE.append(reward/100)
        X_.append(s_[0])
        Theta_.append(s_[2])
        X_PRE.append(s_pre[0])
        Theta__PRE.append(s_pre[2])
        R_DREAM.append(r_pre)

    # if min_loss_r <0.01:
    #     plot = True
    #
    # if plot:
    #     plt.plot(step, X_, 'r', step, X_PRE, 'r--')
    #     plt.plot(step, Reward, 'b', step, R_PRE, 'b--')
    #     plt.plot(step, R_DREAM,'k--')
    #     plt.draw()
    #     plt.pause(0.0000000000000000000000001)
    # plt.close()
    # LR_D *= .99995
    # LR_R *= .99995
    print('Episode:', i, 'Minimum loss S :',min_loss_s, 'Minimum loss R :',min_loss_r,'LR_D :',LR_D,'LR_R :',LR_R,loss_r)
print('Running time: ', time.time() - t1)
