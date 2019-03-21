import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from ENV_PPO_V0 import CartPoleEnv_adv
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#####################  hyper parameters  ####################

MAX_EPISODES = 500000
MAX_EP_STEPS =2500
LR_A = 0.000001    # learning rate for actor
# LR_A = 0.00001   # learning rate for actor
# LR_A = 0.00002   # learning rate for actor
# LR_A = 0.0001    # learning rate for actor
#
LR_C = 0.04    # learning rate for critic
LR_L = 0.001    # learning rate for Lyapunov
# LR_L = 0.00005   # learning rate for Lyapunov
GAMMA = 0.99    # reward discount
labda=10.
tol = 0.001
BATCH_SIZE = 16
Epsilon_pi = 0.1
RENDER = True
METHOD = [
    dict(name='kl_pen', kl_target=5, lam=0.0000005),   # KL penalty
    dict(name='clip', epsilon=0.1),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


env = CartPoleEnv_adv()
env = env.unwrapped
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 5
L_UPDATE_STEPS = 5
#
print(LR_A,LR_C,METHOD['epsilon'],A_UPDATE_STEPS,C_UPDATE_STEPS,LR_L,BATCH_SIZE)

EWMA_p=0.8
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
EWMA_c_loss=np.zeros((1,MAX_EPISODES+1))
EWMA_l_loss=np.zeros((1,MAX_EPISODES+1))
c_loss=1000
###############################  PPO  ####################################

class PPO(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,


        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.tfdc_l = tf.placeholder(tf.float32, [None, 1], 'discounted_l')
        self.l_r = tf.placeholder(tf.float32, [None, 1], 'l_r')
        self.LR_A= tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.v = self._build_c(self.tfs, trainable=True)
        self.l = self._build_l(self.tfs)
        self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
        self.labda = tf.placeholder(tf.float32, None, 'Labda')

        # l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')

        # ACTOR
        pi, pi_params = self._build_a(self.tfs,'pi', trainable=True)
        oldpi, oldpi_params = self._build_a(self.tfs,'oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # self.cons_l = self._build_l(self.cons_S,reuse=True)
        self.l_ = self._build_l(self.S_, reuse=True)

        ALPHA3 = 0.1

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.l_lambda = tf.reduce_mean(ratio * self.l_ - self.l + ALPHA3 * self.l_r) + \
                                GAMMA * Epsilon_pi/(1-GAMMA) * tf.sqrt(2*self.kl_mean)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))+self.labda * self.l_lambda
            else:   # clipping method, find this is better
                self.l_lambda = tf.reduce_mean(ratio * self.l_ - self.l + ALPHA3 * self.l_r)
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))+self.labda * self.l_lambda

        with tf.variable_scope('atrain'):
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.aloss)

        # CRITIC
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.closs)

        #Lyapunov
        self.ladvantage = self.tfdc_l - self.l
        self.lloss = tf.reduce_mean(tf.square(self.ladvantage))
        self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.lloss)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "Model/PPO_Lyapunov_V3.ckpt")  # 1 0.1 0.5 0.001

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -20, 20)

    def update(self, s, a, r,dcl_r,l_r,s_,LR_A,LR_C,LR_L,labda):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam'],self.LR_A: LR_A,self.LR_C: LR_C,self.l_r:l_r,self.S_:s_,self.labda:labda})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)A
            [self.sess.run(self.atrain, {self.tfs: s, self.tfa: a, self.tfadv: adv,self.LR_A: LR_A,self.labda:labda,
                                         self.S_:s_,self.l_r:l_r}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain, {self.tfs: s, self.tfdc_r: r,self.LR_C: LR_C}) for _ in range(C_UPDATE_STEPS)]
        # update Lyapunov
        [self.sess.run(self.ltrain, {self.tfs: s, self.tfdc_l: dcl_r,self.LR_L: LR_L}) for _ in
         range(L_UPDATE_STEPS)]
        return self.sess.run(self.closs,
                             {self.tfs: s, self.tfdc_r: r}),\
               self.sess.run(self.lloss,
                  {self.tfs: s, self.tfdc_l: dcl_r}),\
               self.sess.run(self.l_lambda, {self.tfs: s,self.tfa:a,
                                             self.S_: s_, self.l_r: l_r}), \

    #action 选择模块也是actor模块
    def _build_a(self, s,name, trainable):
        with tf.variable_scope(name):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            # net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            net_3 = tf.layers.dense(net_0, 128, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = 20*tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # a=tf.multiply(mu, self.a_bound, name='scaled_a')
            sigma = tf.layers.dense(net_3, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    #critic模块
    def _build_c(self, s,trainable):
        with tf.variable_scope('Critic'):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l0', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_1, 1, trainable=trainable)  # V(s)
    #Lyapunov
    def _build_l(self, s,reuse=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l0', trainable=trainable)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            # net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_1, 1, trainable=trainable)  # V(s)


    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def get_l(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.l, {self.tfs: s})[0, 0]

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/PPO_Lyapunov_V4.ckpt")
        print("Save to path: ", save_path)


###############################  training  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ppo = PPO(a_dim, s_dim, a_bound)
t1 = time.time()
max_reward=200000
max_ewma_reward=100000
max_step=10

critic_error=200000
lyapnov_error=70000
EWMA_c_loss[0,0]=300000
EWMA_l_loss[0,0]=100000
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    buffer_s, buffer_a, buffer_r,buffer_l,buffer_s_ = [], [], [],[],[]
    ep_reward = 0
    # MAX_EP_STEPS = min(max(500,MAX_EPISODES),1000)
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ppo.choose_action(s)
        # print(a)
        s_, r, done, hit = env.step(a)

        r1 = max(abs(s_[0]) / 5, 3.5 / 5)
        r2 = (abs(s_[2]) / env.theta_threshold_radians)

        l_r = (20 * r1) ** 2 + (20 * r2) ** 2
        # print(a, s, r,s_)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r/10)  # normalize reward, find to be useful
        buffer_l.append(l_r/10)
        buffer_s_.append(s_)
        s = s_

        ep_reward += r

        # update ppo
        if (j + 1) % BATCH_SIZE == 0 or j == MAX_EP_STEPS - 1 or done:
            v_s_ = ppo.get_v(s_)
            l_s_ = ppo.get_l(s_)
            discounted_r = []
            discounted_l = []

            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            for l_r in buffer_l[::-1]:
                l_s_ = l_r + GAMMA * l_s_
                discounted_l.append(l_s_)

            discounted_r.reverse()
            discounted_l.reverse()

            bs, ba, bdcr, bdclr, blr, bs_ = np.vstack(buffer_s), np.vstack(buffer_a), \
                                            np.array(discounted_r)[:, np.newaxis], np.array(discounted_l)[:,
                                                                                   np.newaxis], \
                                            np.vstack(buffer_l), np.vstack(buffer_s_)
            buffer_s, buffer_a, buffer_r, buffer_l, buffer_s_ = [], [], [], [], []
            c_loss, lloss, l_q = ppo.update(bs, ba, bdcr, bdclr, blr, bs_, LR_A, LR_C, LR_L, labda)

            if l_q > tol:
                if labda == 0:
                    labda = 1e-8
                labda = min(labda * 2, 1e2)
            if l_q < -tol:
                labda = labda / 2
        if j == MAX_EP_STEPS - 1:
            # BATCH_SIZE = 64
            a=np.random.randint(1,9)
            BATCH_SIZE = a*8
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            EWMA_c_loss[0, i + 1] = EWMA_p * EWMA_c_loss[0, i] + (1 - EWMA_p) * c_loss
            EWMA_l_loss[0, i + 1] = EWMA_p * EWMA_l_loss[0, i] + (1 - EWMA_p) * lloss
            print('Episode:', i, ' Reward: %i' % int(ep_reward),"Lambda",labda,"Critic loss",EWMA_c_loss[0,i+1],"Lyapunov loss",EWMA_l_loss[0, i + 1],"good","Batch Size",BATCH_SIZE,"EWMA_step = ",EWMA_step[0,i+1],"EWMA_reward = ",EWMA_reward[0,i+1],"LR_A = ",LR_A,"LR_C = ",LR_C,"LR_L = ",LR_L,'Running time: ', time.time() - t1)
            if EWMA_reward[0,i+1]>max_ewma_reward:
                max_ewma_reward=min(EWMA_reward[0,i+1]+1000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                LR_L *= .8  # learning rate for critic
                ppo.save_result()

            if ep_reward> max_reward:
                max_reward = min(ep_reward+5000,500000)
                LR_A *= .8  # learning rate for actor
                LR_C *= .8  # learning rate for critic
                LR_L *= .8  # learning rate for critic
                ppo.save_result()
                print("max_reward : ",ep_reward)

            if EWMA_l_loss[0, i + 1]<lyapnov_error:
                    lyapnov_error = EWMA_l_loss[0, i + 1]-2000
                    LR_L *= 0.8
                    # LR_L *= 0.9


            if EWMA_c_loss[0,i+1]<critic_error:
                critic_error=EWMA_c_loss[0,i+1]-2000
                LR_C *=0.8
                # LR_C *= 0.9


            LR_A *= .999
            LR_C *= .99
            LR_L *= .99
            break

        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            EWMA_c_loss[0,i+1] = EWMA_c_loss[0,i]
            EWMA_l_loss[0, i + 1] = EWMA_l_loss[0,i]
            BATCH_SIZE = min(max(int(EWMA_step[0, i + 1] / 10), 16), 64)
            if hit==1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),"Lambda",labda,"Critic loss",EWMA_c_loss[0,i+1],"Lyapunov loss",EWMA_l_loss[0, i + 1], "break in : ", j, "due to ",
                      "hit the wall", "EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,"LR_C = ",LR_C,"LR_L = ",LR_L,"Batch Size",BATCH_SIZE,'Running time: ', time.time() - t1)
            else:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),"Lambda",labda, "Critic loss",EWMA_c_loss[0,i+1],"Lyapunov loss",EWMA_l_loss[0, i + 1], "break in : ", j, "due to",
                      "fall down","EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,"LR_C = ",LR_C,"LR_L = ",LR_L,"Batch Size",BATCH_SIZE,'Running time: ', time.time() - t1)
            break

print('Running time: ', time.time() - t1)
