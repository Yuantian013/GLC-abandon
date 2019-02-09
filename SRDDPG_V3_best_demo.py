import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from cartpole_uncertainty import CartPoleEnv_adv as dreamer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS =2500
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
labda=10.
RENDER = True
tol = 0.001
# ENV_NAME = 'CartPole-v2'
env = dreamer()
# env = gym.make(ENV_NAME)
env = env.unwrapped


EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_A= tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.labda= tf.placeholder(tf.float32, None, 'Lambda')

        self.a = self._build_a(self.S,)# 这个网络用于及时更新参数
        self.q = self._build_c(self.S, self.a, )# 这个网络是用于及时更新参数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        a_cons = self._build_a(self.S_, reuse=True)
        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        # q_ = self._build_c(self.S_, tf.stop_gradient(a_), reuse=True, custom_getter=ema_getter)
        q_ = self._build_c(self.S_, tf.stop_gradient(a_), reuse=True, custom_getter=ema_getter)
        self.q_cons = self._build_c(self.S_, a_cons, reuse=True)

        self.q_lambda =tf.reduce_mean(self.q - self.q_cons)
        # self.q_lambda = tf.reduce_mean(self.q_cons - self.q)

        a_loss = - tf.reduce_mean(self.q) + self.labda * self.q_lambda  # maximize the q

        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)#以learning_rate去训练，方向是minimize loss，调整列表参数，用adam

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=c_params)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, "Save/cartpole_g10_M1_m0.1_l0.5_tau_0.02.ckpt")  # 1 0.1 0.5 0.001

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self,LR_A,LR_C,labda):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs, self.S_: bs_, self.LR_A: LR_A,self.labda:labda})
        self.sess.run(self.ctrain,{self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.LR_C: LR_C,self.labda:labda})
        return self.sess.run(self.q_lambda,{self.S: bs, self.a: ba, self.R: br, self.S_: bs_}),self.sess.run(self.R, {self.R: br})

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

    def save_result(self):
        # save_path = self.saver.save(self.sess, "Save/cartpole_g10_M1_m0.1_l0.5_tau_0.02.ckpt")
        save_path = self.saver.save(self.sess, "Model/SRDDPG_V3_.ckpt")
        # save_path = self.saver.save(self.sess, "Model/SRDDPG_INITIAL.ckpt")
        print("Save to path: ", save_path)


###############################  training  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.save_result()
var = 5  # control exploration
t1 = time.time()
max_reward=400000
max_ewma_reward=200000
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    ep_reward = 0
    # MAX_EP_STEPS = min(max(500,MAX_EPISODES),1000)
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
        #if var<0.01:
            #a=np.clip(np.random.normal(a, a_bound), -a_bound, a_bound)
        s_, r, done, hit = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .999995    # decay the action randomness
            #var = np.max([var,0.1])
            # LR_A *= .99995
            # LR_C *= .99995
            l_q,l_r=ddpg.learn(LR_A,LR_C,labda)
            # print(l_q,l_r)
            if l_q>tol:
                if labda==0:
                    labda = 1e-8
                labda = min(labda*2,11)
                if labda==11:
                    labda = 1e-8
            if l_q<-tol:
                labda = labda/2
            # lamda_=l_q#
            # labda = max(labda+0.0001*lamda_,0)

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            #EWMA[0,i+1]=EWMA[0,i+1]/(1-(EWMA_p **(i+1)))
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,"good","EWMA_step = ",EWMA_step[0,i+1],"EWMA_reward = ",EWMA_reward[0,i+1],"LR_A = ",LR_A,'lambda',labda,'Running time: ', time.time() - t1)
            # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, 'EWMA: %.2f' % EWMA[0, i + 1])
            # if ep_reward > -20:RENDER = True # 当 回合总 reward 大于 300 时显示模拟窗口
            #if var <= 0.1:RENDER = True # 当 回合总 reward 大于 300 时显示模拟窗口
            #if i > 4990: RENDER = True  # 当 回合总 reward 大于 300 时显示模拟窗口
            # if win>winmax:
            #     ddpg.save_result()
            #     winmax = win
            #     LR_A *= .9  # learning rate for actor
            #     LR_C *= .95  # learning rate for critic
            #     ddpg.save_result()
            # break
            if EWMA_reward[0,i+1]>max_ewma_reward:
                max_ewma_reward=min(EWMA_reward[0,i+1]+1000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()

            if ep_reward> max_reward:
                max_reward = min(ep_reward+5000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()
                print("max_reward : ",ep_reward)
            else:
                LR_A *= .99
                LR_C *= .99
            break

        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            #EWMA[0,i+1]=EWMA[0,i+1]/(1-(EWMA_p **(i+1)))
            if hit==1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, "break in : ", j, "due to ",
                      "hit the wall", "EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,'lambda',labda,'Running time: ', time.time() - t1)
            else:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, "break in : ", j, "due to",
                      "fall down","EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,'lambda',labda,'Running time: ', time.time() - t1)
            break

print('Running time: ', time.time() - t1)
