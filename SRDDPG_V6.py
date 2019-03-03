# Useful Package
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import time
from cartpole_disturb_with_target import CartPoleEnv_adv as dreamer
import os
import math
# For GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#####################  备忘  ###################
#Lambda更新
#Disturb训练方式
#####################  hyper parameters  ####################
MAX_EPISODES = 50000
MAX_EP_STEPS =2500
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
LR_D = 0.0001    # learning rate for disturb
GAMMA = 0.99    # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
CONS_MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
labda=10.
tol = 0.001
MIU = 10.
ALPHA3 = .1
# Function switch
RENDER  = True
DISTURB = False
DREAMER = False

print("Dreamer = ",DREAMER,",DISTURB = " ,DISTURB,",RENDER = ",RENDER)

# For analyse
EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))

# Training setting
var = 5  # control exploration
t1 = time.time()

min_reward=100
min_ewma_reward=50

###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,disturb_switch):
        ###############################  Model parameters  ####################################
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 4), dtype=np.float32)
        self.cons_memory = np.zeros((CONS_MEMORY_CAPACITY, s_dim * 2 + a_dim + 4), dtype=np.float32)
        self.pointer = 0
        self.cons_pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.cons_S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.l_R = tf.placeholder(tf.float32, [None, 1], 'l_r')  # 给lyapunov设计的reward
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.LR_D = tf.placeholder(tf.float32, None, 'LR_D')
        self.labda = tf.placeholder(tf.float32, None, 'Lambda')
        self.a = self._build_a(self.S, )  # 这个网络用于及时更新参数

        self.d = self._build_d(self.S, )  # 这个网络用于及时更新参数
        self.q = self._build_c(self.S, self.a, self.d)  # 这个网络是用于及时更新参数
        self.l = self._build_l(self.S, self.a)   # lyapunov 网络
        self.DISTURB=disturb_switch
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disturber')
        l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')


        ###############################  Model Learning Setting  ####################################

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        target_update = [ema.apply(a_params), ema.apply(c_params), ema.apply(d_params), ema.apply(l_params)]  # soft update operation


        beta = 0.01
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        cons_a = self._build_a(self.cons_S, reuse=True)
        cons_a_ = self._build_a(self.cons_S_, reuse=True)
        d_ = self._build_d(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        # self.cons_d = self._build_d(self.cons_S, reuse=True)
        # cons_d_ = self._build_d(self.cons_S_, reuse=True)

        q_ = self._build_c(self.S_, tf.stop_gradient(a_), tf.stop_gradient(d_), reuse=True, custom_getter=ema_getter)
        l_ = self._build_l(self.S_, tf.stop_gradient(a_), reuse=True, custom_getter=ema_getter)  # lyapunov 网络
        self.cons_l = self._build_l(self.cons_S, tf.stop_gradient(cons_a), reuse=True)
        self.cons_l_ = self._build_l(self.cons_S_, cons_a_, reuse=True)

        self.l_lambda = tf.reduce_mean(self.cons_l_ - self.cons_l + ALPHA3 * self.l_R)
        a_pre_loss = - tf.reduce_mean(self.q)
        a_loss = self.labda * self.l_lambda + tf.reduce_mean(self.q)
        d_loss = tf.reduce_mean(self.q)
        self.apretrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_pre_loss, var_list=a_params)
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)#以learning_rate去训练，方向是minimize loss，调整列表参数，用adam
        self.dtrain = tf.train.AdamOptimizer(self.LR_D).minimize(d_loss,
                                                                 var_list=d_params)  # 以learning_rate去训练，方向是minimize loss，调整列表参数，用adam

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_ + beta*tf.matmul(self.d, tf.transpose(self.d))
            l_target = self.l_R + GAMMA * l_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error, var_list=c_params)
            self.ltrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.l_error, var_list=l_params)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def choose_disturb(self, s):
        return self.sess.run(self.d, {self.S: s[np.newaxis, :]})[0]

    def learn(self,LR_A,LR_D,LR_C,labda):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bd = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim+2]
        br = bt[:, -self.s_dim - 2: -self.s_dim-1]
        blr = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        indices = np.random.choice(CONS_MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.cons_memory[indices, :]
        cons_bs = bt[:, :self.s_dim]
        cons_bs_ = bt[:, -self.s_dim:]
        cons_blr = bt[:, -self.s_dim - 1: -self.s_dim]
        
        if self.DISTURB:
           self.sess.run(self.dtrain, {self.S: bs, self.LR_D: LR_D})

        self.sess.run(self.atrain, {self.S: bs, self.S_: bs_, self.LR_A: LR_A,self.labda: labda,self.cons_S:cons_bs,
                                    self.cons_S_:cons_bs_, self.l_R:cons_blr})
        self.sess.run(self.ctrain,
                      {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.LR_C: LR_C, self.d: bd})
        self.sess.run(self.ltrain,
                      {self.S: bs, self.a: ba, self.S_:bs_, self.l_R: blr, self.LR_C: LR_C})
        return self.sess.run(self.l_lambda, {self.cons_S:cons_bs,
                                    self.cons_S_:cons_bs_, self.l_R:cons_blr}), \
               self.sess.run(self.td_error,
                             {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.LR_C: LR_C, self.d: bd}), \
               self.sess.run(self.l_error, {self.S: bs, self.a: ba, self.S_:bs_, self.l_R: blr})

    def pre_learn(self, LR_A, LR_D, LR_C):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bd = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 2]
        br = bt[:, -self.s_dim - 2: -self.s_dim - 1]
        blr = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        if self.DISTURB:
            self.sess.run(self.dtrain, {self.S: bs, self.LR_D: LR_D})

        self.sess.run(self.apretrain, {self.S: bs, self.S_: bs_, self.LR_A: LR_A})
        self.sess.run(self.ctrain,
                      {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.LR_C: LR_C, self.d: bd})
        self.sess.run(self.ltrain,
                      {self.S: bs, self.a: ba, self.S_: bs_, self.l_R: blr, self.LR_C: LR_C})
        return self.sess.run(self.td_error,
                             {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.LR_C: LR_C, self.d: bd}), \
               self.sess.run(self.l_error, {self.S: bs, self.a: ba, self.S_: bs_, self.l_R: blr})

    def store_transition(self, s, a, d, r, l_r, s_):
        transition = np.hstack((s, a, d,[r], [l_r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def store_edge_transition(self, s, a, d, r, l_r, s_):
        """把数据存入constraint buffer"""
        transition = np.hstack((s, a, d, [r], [l_r], s_))
        index = self.pointer % CONS_MEMORY_CAPACITY  # replace the old memory with new memory
        self.cons_memory[index, :] = transition
        self.cons_pointer += 1

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
    def _build_c(self, s, a,d,reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            w1_d = tf.get_variable('w1_d', [self.s_dim/2, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a)+tf.matmul(d, w1_d)+b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable)
            # net_4 = tf.layers.dense(net_3, 64, activation=tf.nn.relu, name='l5', trainable=trainable)
            return tf.layers.dense(net_3, 1, trainable=trainable)  # Q(s,a)

    # lyapunov模块
    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 512#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a)+b1)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable)
            return tf.layers.dense(net_3, 1, trainable=trainable)  # Q(s,a)

    def _build_d(self, s, reuse=None, custom_getter=None):
        theta_threshold_radians = 20 * 2 * math.pi / 360
        x_threshold = 5
        trainable = True
        with tf.variable_scope('Disturber', reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(s, 512, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 512, activation=tf.nn.relu, name='l3', trainable=trainable)
            net_3 = tf.layers.dense(net_2, 256, activation=tf.nn.relu, name='l4', trainable=trainable)
            d = tf.layers.dense(net_3, self.s_dim/2, activation=tf.nn.tanh, name='d', trainable=trainable)
            return tf.multiply(d, [x_threshold/500,theta_threshold_radians/500], name='scaled_d')

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/SRDDPG_In_Dream_WITHOUT_D.ckpt")
        print("Save to path: ", save_path)
###############################  DREAMER  ####################################

class Dreamer(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
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

        #Render Part
        self.viewer = None
        self.state = None
        self.x_threshold=5
        #Learning Part
        self.dreamer = self._build_dreamer(self.S, self.A,self.S_L) #S_=linear_model+DNN=S_L+DNN(S,A)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dreamer')

        self.dreamer_loss_s = tf.reduce_mean(tf.squared_difference(self.S_ , self.dreamer))

        self.dreamertrain_s = tf.train.AdamOptimizer(self.LR_D).minimize(self.dreamer_loss_s,var_list = d_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/SRDDPG_Dreamer_V1.ckpt")  # 1 0.1 0.5 0.001

    def dream(self, s,a,d):
        # self.gravity = np.random.normal(10, 0.1)
        # self.masscart = np.random.normal(1, 0.1)
        # self.masspole = np.random.normal(0.1, 0.01)
        x, x_dot, theta, theta_dot = s
        force = a[0]
        costheta = 1
        sintheta = theta
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x_ = x + self.tau * x_dot+d[0]
        x_dot_ = x_dot + self.tau * xacc
        theta_ = theta + self.tau * theta_dot+d[1]
        theta_dot_ = theta_dot + self.tau * thetaacc
        s_linear = np.array([x_, x_dot_, theta_, theta_dot_])

        s_=self.sess.run(self.dreamer, {self.S: s[np.newaxis, :],self.A: a[np.newaxis, :],self.S_L: s_linear[np.newaxis, :]})[0]

        x, _, theta, _ = s_
        r_1 = ((1 - abs(x)))
        r_2 = (((20 * 2 * math.pi / 360) / 4) - abs(theta)) / ((20 * 2 * math.pi / 360) / 4)
        reward = np.sign(r_2) * ((10 * r_2) ** 2) + np.sign(r_1) * ((10 * r_1) ** 2)
        self.state = s_
        return s_,reward

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
env = dreamer()
env = env.unwrapped
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
env_dream=Dreamer(a_dim, s_dim, a_bound)
ddpg = DDPG(a_dim, s_dim, a_bound,DISTURB)
###############################  TRAINING  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    REWARD = 0
    l_loss = np.nan
    c_loss = np.nan
    for j in range(MAX_EP_STEPS):


        #Visulization
        if RENDER:
            if DREAMER:
                env_dream.render()
            else:
                env.render()

        # Choose action
        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration


        if DISTURB:
            # Choose disturb
            # Add exploration noise
            d = ddpg.choose_disturb(s)
            d = np.random.normal(d, abs(d * 0.02 * var))  # add randomness to disturb selection for exploration
        else:
            d=[0,0]


        # RUN IN REAL IN TO GET INFORMATION OF DIE OR NOT
        # IF Dreamer_update=True, GET INFORMATION OF THE S,A,R1,S_
        # 得到的是真实的 s,a->s_ 和 r
        # 主要是判断是否游戏结束
        s_, r, done, hit = env.step(a,d)             # S_=ENV(S,A), R=REWARD(S_)
        l_r = np.square(5*s_[0]/env.x_threshold) + np.square(10*s_[2]/env.theta_threshold_radians)
        # l_r = np.linalg.norm(s_,2)
        # RUN IN DREAM
        # 得到的是梦境中的s,a->s_dream 和 r_dream
        # 如果在梦境，那么s_next 和 reward 就被梦境值覆盖
        s_next=s_
        reward=r
        if DREAMER:
            s_next, reward = env_dream.dream(s,a,d)


        #储存s,a和s_next,reward用于DDPG的学习
        ddpg.store_transition(s, a, d,(reward / 10), l_r, s_next)

        #如果状态接近边缘 就存储到边缘memory里
        if np.abs(s[0]) > env.x_threshold*0.8 or np.abs(s[2]) > env.theta_threshold_radians*0.8:
            ddpg.store_edge_transition(s, a, d, (reward / 10), l_r, s_next)
        # ddpg.store_edge_transition(s, a, d, (reward / 10), l_r, s_next)
        #DDPG LEARN

        # if ddpg.pointer > MEMORY_CAPACITY and ddpg.cons_pointer <= CONS_MEMORY_CAPACITY:
        #     var *= .99999
        #     c_loss, l_loss = ddpg.pre_learn(LR_A, LR_C, LR_D)

        if ddpg.pointer > MEMORY_CAPACITY and ddpg.cons_pointer > CONS_MEMORY_CAPACITY:
            # Decay the action randomness
            var *= .99999
            l_q,c_loss, l_loss=ddpg.learn(LR_A,LR_C,LR_D,labda)
            if l_q>tol:
                if labda==0:
                    labda = 1e-8
                labda = min(labda*2,1e8)
                if labda==1e8:
                    labda = 1e-8
            if l_q<-tol:
                labda = labda/2

        # 梦境状态更新
        s = s_next

        # 现实状态与梦境同步，用于进行下一次的现实STEP
        env.state = s_next

        # 计算总得分
        REWARD += reward

        # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
        if j == MAX_EP_STEPS - 1:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*REWARD
            print('Episode:', i, ' Reward: %.1f' % REWARD,'Explore: %.2f' % var,"good",
                  "EWMA_step = ",int(EWMA_step[0,i+1]),"EWMA_reward = ",EWMA_reward[0,i+1],"LR_A = ",LR_A,'lambda',labda,
                  'LR_D :',LR_D, 'lyapunov_error:', l_loss , 'critic_error:', c_loss )
            if EWMA_reward[0,i+1]<min_ewma_reward:
                min_ewma_reward=EWMA_reward[0,i+1]+1000
                LR_A *= .8  # learning rate for actor
                LR_D *= .8  # learning rate for disturb
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()

            if REWARD< min_reward:
                max_reward = REWARD
                LR_A *= .8  # learning rate for actor
                LR_D *= .8  # learning rate for disturb
                LR_C *= .8  # learning rate for critic
                ddpg.save_result()
                print("min_reward : ",REWARD)
            else:
                LR_A *= .99
                LR_D *= .99
                LR_C *= .99
            break
        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*REWARD

            if hit==1:
                print('Episode:', i,  ' Reward: %.1f' % REWARD, 'Explore: %.2f' % var, "break in : ", j, "due to ",
                      "hit the wall", "EWMA_step = ",int(EWMA_step[0,i+1]), "EWMA_reward = ", EWMA_reward[0, i + 1],
                      "LR_A = ",LR_A,'lambda',labda,'LR_D :',LR_D, 'lyapunov_error:', l_loss, 'critic_error:', c_loss)
            else:
                print('Episode:', i,  ' Reward: %.1f' % REWARD,'Explore: %.2f' % var, "break in : ", j, "due to",
                      "fall down","EWMA_step = ",int(EWMA_step[0,i+1]), "EWMA_reward = ", EWMA_reward[0, i + 1],
                      "LR_A = ",LR_A,'lambda',labda,'LR_D :',LR_D, 'lyapunov_error:', l_loss, 'critic_error:', c_loss)
            break