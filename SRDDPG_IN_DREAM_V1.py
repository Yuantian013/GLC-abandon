import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from cartpole_uncertainty import CartPoleEnv_adv as dreamer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS =2500
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
LR_D = 0.00001   # learning rate for DREAM STATE
LR_R = 0.00001   # learning rate for DREAM SCORE
GAMMA = 0.99    # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
labda=10.
tol = 0.001

RENDER = True
Dreamer_update=False
REAL_SCORE=True
print("Dreamer_update = ",Dreamer_update," , REAL_SCORE = ",REAL_SCORE)
env = dreamer()
env = env.unwrapped

EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))

var = 5  # control exploration
t1 = time.time()
min_loss_s=0.001
min_loss_r=0.01
loss_r=100
max_reward=100000
max_ewma_reward=20000

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
        save_path = self.saver.save(self.sess, "Model/SRDDPG_IN_DREAM_0.5.ckpt")
        print("Save to path: ", save_path)
###############################  DREAMER  ####################################
class Dreamer(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.LR_D= tf.placeholder(tf.float32, None, 'LR_D')
        self.LR_R = tf.placeholder(tf.float32, None, 'LR_R')

        self.x_threshold = 5
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.dreamer = self._build_dreamer(self.S, self.A, )
        self.score=self._build_score(self.dreamer)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dreamer')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Score')

        self.viewer = None
        self.state = None


        self.dreamer_loss_s = tf.reduce_mean(tf.squared_difference(self.S_, self.dreamer))

        self.dreamer_loss_r = tf.reduce_mean(tf.squared_difference(self.R, self.score))

        self.dreamertrain_s = tf.train.AdamOptimizer(self.LR_D).minimize(self.dreamer_loss_s,var_list = d_params)

        self.dreamertrain_r = tf.train.AdamOptimizer(self.LR_R).minimize(self.dreamer_loss_r, var_list=r_params)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/SRDDPG_D_V1.ckpt")  # 1 0.1 0.5 0.001
    def dream(self, s,a):
        self.state=self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :]})[0]
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
        save_path = self.saver.save(self.sess, "Model/SRDDPG_D_ONLINE.ckpt")
        print("Save to path: ", save_path)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

###############################  INITIALIZE  ####################################
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

env_dream=Dreamer(a_dim, s_dim, a_bound)
ddpg = DDPG(a_dim, s_dim, a_bound)
###############################  TRAINING  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    DREAM_REWARD = 0
    REAL_REWARD = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env_dream.render()
        # Choose action
        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration

        # RUN IN REAL IN TO GET INFORMATION OF DIE OR NOT
        # IF Dreamer_update=True, GET INFORMATION OF THE S,A,R1,S_
        # 得到的是真实的 s,a->s_ 和 r
        # 主要是判断是否游戏结束
        # r_real是s_real的函数
        s_real, r_real, done, hit = env.step(a)             # S_=ENV(S,A), R=REWARD(S_)


        # RUN IN DREAM
        # 得到的是梦境中的s,a->s_dream 和 r_dream
        s_dream, r_dream = env_dream.dream(s, a)

        # 得分修正
        r_dream=r_dream*100

        # 构建得分函数 R=REWARD(S_) 和真实得分函数一样
        x, _, theta, _=s_dream
        r_1 = ((1 - abs(x)))  # - 0.8
        r_2 = (((20 * 2 * math.pi / 360) / 4) - abs(theta)) / ((20 * 2 * math.pi / 360) / 4)  # - 0.5
        reward = np.sign(r_2) * ((10 * r_2) ** 2) + np.sign(r_1) * ((10 * r_1) ** 2)

        # 两种得分给予方式，真实得分和梦境拟合得分
        if REAL_SCORE==False:
            reward=r_dream

        #储存s,a和梦境中的s_dream,r_dream用于DDPG的学习
        # ddpg.store_transition(s, a, (r/10)[0], s_)
        ddpg.store_transition(s, a, (reward / 10), s_dream)

        #如果DRAMER在线更新，则储存s,a和真实的s_,r
        if Dreamer_update:
            env_dream.store_transition(s, a, r_real / 100, s_real)

        #DDPG LEARN
        if ddpg.pointer > MEMORY_CAPACITY:
            # Decay the action randomness
            var *= .9999
            l_q,l_r=ddpg.learn(LR_A,LR_C,labda)

            if l_q>tol:
                if labda==0:
                    labda = 1e-8
                labda = min(labda*2,11)
                if labda==11:
                    labda = 1e-8
            if l_q<-tol:
                labda = labda/2

        # DREAMER LEARN
        if Dreamer_update:
            if env_dream.pointer > MEMORY_CAPACITY:
                LR_D *= .99995
                LR_R *= .99995
                loss_s, loss_r = env_dream.learn(LR_D, LR_R)
                if loss_s < min_loss_s:
                    LR_D *= (loss_s / min_loss_s)
                    min_loss_s = loss_s
                if loss_r < min_loss_r:
                    LR_R *= (loss_r / min_loss_r)
                    min_loss_r = loss_r
                    env_dream.save_result()

        # 梦境状态更新
        s = s_dream

        # 现实状态与梦境同步，用于进行下一次的现实STEP
        env.state = s_dream

        # 得到现实得分与梦境得分，帮助分析
        DREAM_REWARD += reward
        REAL_REWARD += r_real

        # OUTPUT TRAINING INFORMATION
        if j == MAX_EP_STEPS - 1:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*DREAM_REWARD
            print('Episode:', i, ' Reward: %i' % int(DREAM_REWARD),' REAL Reward: %i' % int(REAL_REWARD), 'Difference : ',abs(1-(DREAM_REWARD/REAL_REWARD)),'Explore: %.2f' % var,"good","EWMA_step = ",EWMA_step[0,i+1],"EWMA_reward = ",EWMA_reward[0,i+1],"LR_A = ",LR_A,'lambda',labda,'Minimum loss S :',min_loss_s, 'Minimum loss R :',min_loss_r,'LR_D :',LR_D,'LR_R :',LR_R,loss_r)
            if EWMA_reward[0,i+1]>max_ewma_reward:
                max_ewma_reward=min(EWMA_reward[0,i+1]+1000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()

            if DREAM_REWARD> max_reward:
                max_reward = min(DREAM_REWARD+5000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()
                print("max_reward : ",DREAM_REWARD)
            else:
                LR_A *= .99
                LR_C *= .99
            break
        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*DREAM_REWARD

            if hit==1:
                print('Episode:', i, ' Reward: %i' % int(DREAM_REWARD),' REAL Reward: %i' % int(REAL_REWARD),'Difference : ',abs(1-(DREAM_REWARD/REAL_REWARD)), 'Explore: %.2f' % var, "break in : ", j, "due to ",
                      "hit the wall", "EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,'lambda',labda, 'Minimum loss S :',min_loss_s, 'Minimum loss R :',min_loss_r,'LR_D :',LR_D,'LR_R :',LR_R,loss_r)
            else:
                print('Episode:', i, ' Reward: %i' % int(DREAM_REWARD), ' REAL Reward: %i' % int(REAL_REWARD),'Difference : ',abs(1-(DREAM_REWARD/REAL_REWARD)),'Explore: %.2f' % var, "break in : ", j, "due to",
                      "fall down","EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,'lambda',labda, 'Minimum loss S :',min_loss_s, 'Minimum loss R :',min_loss_r,'LR_D :',LR_D,'LR_R :',LR_R,loss_r)
            break




