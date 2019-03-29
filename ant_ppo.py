import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#####################  hyper parameters  ####################

MAX_EPISODES = 50000000
MAX_EP_STEPS =2048
LR_A = 0.0000001    # learning rate for actor
LR_C = 0.00002    # learning rate for critic

GAMMA = 0.99    # reward discount
BATCH_SIZE = 32
RENDER = False
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

ENV_NAME = 'Ant-v2'

env = gym.make(ENV_NAME)
env = env.unwrapped
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 5

print(LR_A,LR_C,METHOD['epsilon'],A_UPDATE_STEPS,C_UPDATE_STEPS,BATCH_SIZE)

EWMA_p=0.95
EWMA_step=np.zeros((1,MAX_EPISODES+1))
EWMA_reward=np.zeros((1,MAX_EPISODES+1))
iteration=np.zeros((1,MAX_EPISODES+1))
EWMA_c_loss=np.zeros((1,MAX_EPISODES+1))
c_loss=1000
###############################  PPO  ####################################

class PPO(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,


        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.LR_A= tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.v = self._build_c(self.tfs, trainable=True)


        # ACTOR
        pi, pi_params = self._build_a(self.tfs,'pi', trainable=True)
        oldpi, oldpi_params = self._build_a(self.tfs,'oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.aloss)

        # CRITIC
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.closs)


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -a_bound, a_bound)

    def update(self, s, a, r,LR_A,LR_C):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam'],self.LR_A: LR_A,self.LR_C: LR_C})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)A
            [self.sess.run(self.atrain, {self.tfs: s, self.tfa: a, self.tfadv: adv,self.LR_A: LR_A,self.LR_C: LR_C}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain, {self.tfs: s, self.tfdc_r: r,self.LR_A: LR_A,self.LR_C: LR_C}) for _ in range(C_UPDATE_STEPS)]
        return self.sess.run(self.closs,
                             {self.tfs: s, self.tfdc_r: r})

    #action 选择模块也是actor模块
    def _build_a(self, s,name, trainable):
        with tf.variable_scope(name):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = a_bound*tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
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
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net_3, 1, trainable=trainable)  # V(s)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_result(self):
        save_path = self.saver.save(self.sess, "Model/ant_ppo.ckpt")
        print("Save to path: ", save_path)


###############################  training  ####################################
# env.seed(1)   # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子
#models.py mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
# print(env.action_space.high)
# print(env.action_space.low)
ppo = PPO(a_dim, s_dim, a_bound)
t1 = time.time()
max_reward=400
max_ewma_reward=200
max_step=10

critic_error=500
EWMA_c_loss[0,0]=1000
for i in range(MAX_EPISODES):
    iteration[0,i+1]=i+1
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ppo.choose_action(s)
        # print(a)
        s_, r, done,_ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)
        s = s_

        ep_reward += r
        # update ppo
        if (j + 1) % BATCH_SIZE == 0 or j == MAX_EP_STEPS - 1 or done:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            c_loss=ppo.update(bs, ba, br,LR_A,LR_C)
        if j == MAX_EP_STEPS - 1:
            # a = np.random.randint(1, 9)
            # BATCH_SIZE = a * 8
            # BATCH_SIZE=32
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            EWMA_c_loss[0, i + 1] = EWMA_p * EWMA_c_loss[0, i] + (1 - EWMA_p) * c_loss
            print(max_ewma_reward,'Episode:', i, ' Reward: %i' % int(ep_reward),"Critic loss",EWMA_c_loss[0,i+1],"good","Batch Size",BATCH_SIZE,"EWMA_step = ",EWMA_step[0,i+1],"EWMA_reward = ",EWMA_reward[0,i+1],"LR_A = ",LR_A,"LR_C = ",LR_C,'Running time: ', time.time() - t1)
            if EWMA_reward[0,i+1]>max_ewma_reward:
                max_ewma_reward=EWMA_reward[0,i+1]
                LR_A *= .8  # learning rate for actor
                # LR_C *= .8  # learning rate for critic
                print("fuck")
                ppo.save_result()

            if ep_reward> max_reward:
                max_reward = ep_reward
                LR_A *= .8  # learning rate for actor
                # LR_C *= .8  # learning rate for critic
                ppo.save_result()
                print("max_reward : ",ep_reward)

            if EWMA_c_loss[0,i+1]<critic_error:
                critic_error=EWMA_c_loss[0,i+1]
                LR_C *=0.9


            LR_A *= .99
            LR_C *= .99

            break

        elif done:
            EWMA_step[0,i+1]=EWMA_p*EWMA_step[0,i]+(1-EWMA_p)*j
            EWMA_reward[0,i+1]=EWMA_p*EWMA_reward[0,i]+(1-EWMA_p)*ep_reward
            EWMA_c_loss[0, i + 1] = EWMA_p * EWMA_c_loss[0, i] + (1 - EWMA_p) * c_loss
            # BATCH_SIZE = 32
            print('Episode:', i, ' Reward: %i' % int(ep_reward), "Critic loss",EWMA_c_loss[0,i+1], "break in : ", j, "due to",
                      "fall down","EWMA_step = ", EWMA_step[0, i + 1], "EWMA_reward = ", EWMA_reward[0, i + 1],"LR_A = ",LR_A,"LR_C = ",LR_C,"Batch Size",BATCH_SIZE,'Running time: ', time.time() - t1)
            break

print('Running time: ', time.time() - t1)
