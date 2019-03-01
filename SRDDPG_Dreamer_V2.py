import tensorflow as tf
import numpy as np
import time
from cartpole_uncertainty import CartPoleEnv_adv as real_env
from cartpole_disturb import CartPoleEnv_adv as linear_env
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math
#####################  hyper parameters  ####################
MAX_EPISODES = 50000
MAX_EP_STEPS =500
LR_D = 0.001 # learning rate for Dreamer
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
RENDER = False
env = real_env()
env = env.unwrapped
dream_linear=linear_env()
dream_linear = dream_linear.unwrapped
#暂时去掉了uncertainty模型里的不确定性
var = 5  # control exploration
t1 = time.time()
plot=False
min_loss_s=0.01
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

###############################  DDPG  ####################################
class Dreamer(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        tf.reset_default_graph()
        # Model parameter
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 3 + a_dim), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.S_L = tf.placeholder(tf.float32, [None, s_dim], 's_l')
        self.LR_D= tf.placeholder(tf.float32, None, 'LR_D')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')

        # Dynamics Parameter
        self.gravity = 10
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20
        self.tau = 0.02  # seconds between state updates


        #Learning Part
        self.dreamer = self._build_dreamer(self.S, self.A,self.S_L) #S_=linear_model+DNN=S_L+DNN(S,A)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dreamer')

        self.dreamer_loss_s = tf.reduce_mean(tf.squared_difference(self.S_ , self.dreamer))

        self.dreamertrain_s = tf.train.AdamOptimizer(self.LR_D).minimize(self.dreamer_loss_s,var_list = d_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def dream(self, s,a):
        x, x_dot, theta, theta_dot = s
        force = a[0]
        costheta = 1
        sintheta = theta
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x_ = x + self.tau * x_dot
        x_dot_ = x_dot + self.tau * xacc
        theta_ = theta + self.tau * theta_dot
        theta_dot_ = theta_dot + self.tau * thetaacc
        s_linear = np.array([x_, x_dot_, theta_, theta_dot_])
        return self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :],self.S_L: s_linear[np.newaxis, :]})[0]

    def learn(self,LR_D):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]
        bs_l=bt[:, -self.s_dim - 4: -self.s_dim]
        self.sess.run(self.dreamertrain_s, {self.S: bs,self.A: ba, self.S_: bs_,self.S_L: bs_l, self.LR_D: LR_D})

        return self.sess.run(self.dreamer_loss_s, {self.S: bs,self.A: ba, self.S_: bs_,self.S_L: bs_l, self.LR_D: LR_D})

    def store_transition(self, s, a,s_):
        x, x_dot, theta, theta_dot = s
        force = a[0]
        costheta = 1
        sintheta = theta
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x_ = x + self.tau * x_dot
        x_dot_ = x_dot + self.tau * xacc
        theta_ = theta + self.tau * theta_dot
        theta_dot_ = theta_dot + self.tau * thetaacc
        s_linear = np.array([x_, x_dot_, theta_, theta_dot_])

        transition = np.hstack((s, a,s_linear,s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_dreamer(self, s, a,s_linear,reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Dreamer', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a)+ b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_2, self.s_dim, trainable=trainable)+s_linear


    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/SRDDPG_Dreamer_V2.ckpt")
        print("Save to path: ", save_path)

###############################  Initialize ####################################
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)
dreamer=Dreamer(a_dim, s_dim, a_bound)

###############################  Training  ####################################
for i in range(MAX_EPISODES):
    # Reset real s
    s = env.reset()
    dream_linear.state = s
    # For plot
    X_=[]
    Theta_ = []
    X_PRE=[]
    Theta__PRE = []
    step=[]
    X_L=[]
    Theta_L=[]
    # Load different model for get different situation information
    if int(i % 3) == 0:
        ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_V3.ckpt")  # 1 0.1 0.5 0.001
    elif int(i % 5) == 0:
        ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_IN_DREAM_0.5.ckpt")  # 1 0.1 0.5 0.001
    else:
        ddpg.saver.restore(ddpg.sess, "Model/SRDDPG_INITIAL.ckpt")  # 1 0.1 0.5 0.001

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
        # Get real s,a,s_
        s_, _, done, hit = env.step(a)


        # get the s,a,s_linear for plot
        # 实际上是为了输出同一个套输入力下，真实模型和线性模型的轨迹图像，他们每时每刻的s并不一样
        s_linear_, _, _, _ = dream_linear.step(a)


        # DNN Model s,a,s_pre(for plot)
        # s_pre=linear_s+DNN(s,a)
        s_pre=dreamer.dream(s,a)

        # Save real s,a,s_,s_linear for RL
        dreamer.store_transition(s, a, s_)

        if done:
            break

        # DREAMER LEARN
        if dreamer.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            LR_D*= .99995
            loss_s=dreamer.learn(LR_D)
            if loss_s <min_loss_s:
                LR_D *= (loss_s/min_loss_s)
                min_loss_s=loss_s
                dreamer.save_result()

        # 真实状态更新
        s = s_

        # For plot
        step.append(j)
        X_.append(s_[0])
        Theta_.append(s_[2])
        X_PRE.append(s_pre[0])
        Theta__PRE.append(s_pre[2])
        X_L.append(s_linear_[0])
        Theta_L.append(s_linear_[2])

    if min_loss_s <0.0005:
        plot = True

    if plot:
        plt.plot(step, X_, 'r', step, X_PRE, 'r--')
        plt.plot(step, Theta_, 'b', step, Theta__PRE, 'b--')
        plt.plot(step, X_L, 'g', step, Theta_L, 'k')
        plt.draw()
        plt.pause(5)
    plt.close()

    print('Episode:', i, 'Minimum loss S :',min_loss_s, 'LR_D :',LR_D,)
print('Running time: ', time.time() - t1)
#Best error = 0.0001611716