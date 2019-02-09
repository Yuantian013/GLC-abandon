import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

class QUADROTOR():
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    def __init__(self):

        # crazyflie: physical parameters for the Crazyflie 2.0

        # This function creates a struct with the basic parameters for the
        # Crazyflie 2.0 quad rotor (without camera, but with about 5 vicon
        # markers)
        # Model assumptions based on physical measurements:
        # motor + mount + vicon marker = mass point of 3g
        # arm length of mass point: 0.046m from center
        # battery pack + main board are combined into cuboid (mass 18g) of
        # dimensions:
        # width  = 0.03m
        # depth  = 0.03m
        # height = 0.012m

        m = 0.030 # weight (in kg) with 5 vicon markers (each is about 0.25g)
        gravity = 9.81 #gravitational constant
        I = [[1.43e-5, 0, 0],
                  [0, 1.43e-5, 0],
                  [0, 0, 2.89e-5]] #inertial tensor in m^2 kg
        L = 0.046 # arm length in m

        self.m = m
        self.g = gravity
        self.I = I
        self.invI=np.linalg.inv(self.I)
        self.arm_length = L

        self.max_angle = 40 * math.pi/180 # you can specify the maximum commanded angle here
        self.max_F = 2.5 * self.m * self.g # left these untouched from the nano plus
        self.min_F = 0.05 * self.m * self.g # left these untouched from the nano plus

        # You can add any fields you want in self
        # for example you can add your controller gains by
        # self.k = 0, and they will be passed into controller.

################以下内容还未修改，不过暂时不用到
        self.force_mag = 20
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 5
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_v,
            self.theta_threshold_radians * 2,
            self.max_w])

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def quadEOM(self,t,s, F, M):


        # QUADEOM Solve quadrotor equation of motion
        # quadEOM calculate the derivative of the state vector
        # INPUTS:
        # t      - 1 x 1, time
        # s      - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
        # F      - 1 x 1, thrust output from controller (only used in simulation)
        # M      - 3 x 1, moments output from controller (only used in simulation)
        # self   -  output from init() and whatever parameters you want to pass in
        #  OUTPUTS:
        # sdot   - 13 x 1, derivative of state vector s


        self.A = [[0.25, 0, -0.5 / self.arm_length],
                  [0.25, 0.5 / self.arm_length, 0],
                  [0.25, 0, 0.5 / self.arm_length],
                  [0.25, -0.5 / self.arm_length, 0]]


        prop_thrusts = np.dot(self.A, [[F], M[0], M[1]])

        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, self.max_F / 4), self.min_F / 4)
        B = [[1, 1, 1, 1],
             [0, self.arm_length, 0, -self.arm_length],
             [-self.arm_length, 0, self.arm_length, 0]]

        F = np.dot(B[0],prop_thrusts_clamped)

        M = np.reshape([np.dot(B[1:3],prop_thrusts_clamped)[0],np.dot(B[1:3],prop_thrusts_clamped)[1],M[2]],[3])

        # Assign states
        x = s[0]
        y = s[1]
        z = s[2]
        xdot = s[3]
        ydot = s[4]
        zdot = s[5]
        qW = s[6]
        qX = s[7]
        qY = s[8]
        qZ = s[9]
        p = s[10]
        q = s[11]
        r = s[12]
        quat = [qW,qX,qY,qZ]

        bRw = QuatToRot(quat)
        wRb = bRw.T

        # Acceleration
        accel = 1 / self.m * (np.dot(wRb, [[0],[0], [F]]) - [[0], [0], [self.m * self.g]])
        # Angular velocity
        K_quat = 2 #this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1 - (qW*qW + qX*qX+ qY*qY + qZ*qZ)
        qdot = np.dot(np.multiply( [[0., -p, -q, -r],
                                    [p,  0. ,-r,  q],
                                    [q,  r,  0., -p],
                                    [r, -q,  p,  0.]],-1/2),quat)+np.multiply(K_quat * quaterror,quat)


        # Angular acceleration
        omega = [p,q,r]
        pqrdot   = np.dot(self.invI,(M - np.cross(omega, np.dot(self.I,omega))))

        # Assemble sdot
        sdot = np.zeros([13])
        sdot[0]  = xdot
        sdot[1] = ydot
        sdot[2] = zdot
        sdot[3] = accel[0]
        sdot[4] = accel[1]
        sdot[5] = accel[2]
        sdot[6] = qdot[0]
        sdot[7] = qdot[1]
        sdot[8] = qdot[2]
        sdot[9] = qdot[3]
        sdot[10] = pqrdot[0]
        sdot[11] = pqrdot[1]
        sdot[12] = pqrdot[2]
        return sdot

    def step(self, F,M):
        time = np.linspace(0, 0.01, 2)
        s= self.state
        s_ = odeint(self.quadEOM, s, time, args=(F, M), tfirst=True)
        self.state=s_[1]
        return self.state

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        self.steps_beyond_done = None
        return np.array(self.state)



def stateToQd(x):
    # Converts qd struct used in hardware to x vector used in simulation
    # x is 1 x 13 vector of state variables [pos vel quat omega]
    # qd is a struct including the fields pos, vel, euler, and omega

    # current state
    pos = x[0:3]
    vel = x[3:6]
    Rot = QuatToRot(x[6:10].T)
    [phi, theta, yaw] = RotToRPY_ZXY(Rot)

    euler = [phi,theta,yaw]
    omega = x[10:13]
    return [pos,vel,euler,omega]
def QuatToRot(q):
    # QuatToRot Converts a Quaternion to Rotation matrix
    # normalize q

    q = q/np.sqrt(sum(np.multiply(q,q)))
    qahat=np.zeros([3,3])
    qahat[0, 1] = -q[3]
    qahat[0, 2] = q[2]
    qahat[1, 2] = -q[1]
    qahat[1, 0] = q[3]
    qahat[2, 0] = -q[2]
    qahat[2, 1] = q[1]
    R = np.eye(3) + 2*np.dot(qahat,qahat) + 2*np.dot(q[0],qahat)
    return R
def RPYtoRot_ZXY(phi,theta,psi):

    R = [[math.cos(psi) * math.cos(theta) - math.sin(phi) * math.sin(psi) * math.sin(theta),
          math.cos(theta) * math.sin(psi) + math.cos(psi) * math.sin(phi) * math.sin(theta),
         - math.cos(phi) * math.sin(theta)],
         [- math.cos(phi) * math.sin(psi),
          math.cos(phi) * math.cos(psi),
          math.sin(phi)],
         [math.cos(psi) * math.sin(theta) + math.cos(theta) * math.sin(phi) * math.sin(psi),
          math.sin(psi) * math.sin(theta) - math.cos(psi) * math.cos(theta) * math.sin(phi),
          math.cos(phi) * math.cos(theta)]]
    return R
def RotToRPY_ZXY(R):
    # RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
    # The rotation matrix in this function is world to body [bRw] you will
    # need to transpose the matrix if you have a body to world [wRb] such
    # that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and
    # [wP] is a point in the world frame
    # bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
    #           cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
    #          -cos(phi)*sin(theta)]
    #         [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
    #         [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
    #            sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
    #            cos(phi)*cos(theta)]

    phi = math.asin(R[1, 2])
    psi = math.atan2(-R[1, 0] / math.cos(phi), R[1, 1] / math.cos(phi))
    theta = math.atan2(-R[0, 2] / math.cos(phi), R[2, 2] / math.cos(phi))
    return [phi,theta,psi]


x0=[1.5,12.,144.,12.,14.,15.,12.,1.,13.,15.,13.,12.,1.]
F=2
M=[[3],[2.5],[1]]
env=QUADROTOR()
env.state=x0
s_=env.step(F,M)
print(s_)