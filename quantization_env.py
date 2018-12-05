# encoding: utf-8
import numpy as np
import my_common as mc
from torch.autograd import Variable
import math
import torch

class QuantizationEnv(object):
    #codebook是np格式的矩阵m*n
    #b以01向量进行计算的结果为hard
    #b以浮点数向量进行计算的结果为soft
    #self.action一定是01向量
    def __init__(self, codebook, b, x, code_len, coff):
        self.reward_coff = coff
        self.b = b
        self.x = x
        self.C = codebook
        self.cl = code_len
        #yyj
        self.action_bin=self.b

        self.observation = self.generate_state(self.action_bin)
        self.mask = False
        self.hard_pre_v = 0
        self.hard_c_v = 0
        self.soft_pre_v = 0
        self.soft_c_v = 0
        self.is_first_v = True
        self.reward = 0
        self.history_list = []
        self.info = []

    #还原状态为初始状态
    def reset(self):
        #yyj
        self.action_bin = self.b
        self.observation = self.generate_state(self.action_bin)
        self.mask = False
        self.hard_pre_v = 0
        self.hard_c_v = 0
        self.soft_pre_v = 0
        self.soft_c_v = 0
        self.is_first_v = True
        self.reward = 0

        return self.observation

    #step by yyj
    def step(self, action, control_bit, actor_size):
        #如果得到的b是全0的，则需要重新随机一个b作为action

        self.action_bin[control_bit:control_bit+actor_size] = mc.binazation(action)
        self.action_b[control_bit:control_bit+actor_size] = action.numpy()
        #action_b=self.b
        #action_b[control_bit]=action
        if self.action_bin.sum() == 0:
            #self.action = self.rand_action(self.cl)
            #全0可以看做是出现了一种错误的情况
            self.mask = True

        self.observation = self.compute_Cbx(self.action_bin)
        self.soft_observation=self.compute_Cbx(self.action_b)

        if self.is_first_v:
            #yyj 第一次的时候 pre是用随机b所计算出的cb-x的模
            self.hard_pre_v=self.compute_v(self.compute_Cbx(self.b))
            self.hard_c_v = self.compute_v(self.observation)
            self.soft_pre_v = self.compute_v(self.compute_Cbx(self.b))
            self.soft_c_v = self.compute_v(self.soft_observation)
            self.is_first_v = False
        else:
            self.hard_pre_v = self.hard_c_v
            self.hard_c_v = self.compute_v(self.observation)
            self.soft_pre_v = self.soft_c_v
            self.soft_c_v = self.compute_v(self.soft_observation)
            self.reward = self.compute_reward()

        if self.hard_c_v < 10 or self.hard_c_v > 10000:
            self.mask = True
        #如果两次计算得到的浮点数向量很靠近，说明已经收敛可以跳出了
        if (self.soft_pre_v - self.soft_c_v).sum() < 0.0001:
             self.mask = True
        return self.observation, self.reward, self.mask, self.hard_c_v

    #计算reward
    # 如果当前得到的模长大于上一次得到的模长，说明这个b的走势不好，reward应当给负值
    # 反之，应当是正值
    def compute_reward(self):
        #np.linalg.norm(a,2)所有元素的平方和开更号
        hard = (self.hard_pre_v - self.hard_c_v)/self.hard_pre_v
        soft = (self.soft_pre_v - self.soft_c_v)/self.soft_pre_v
        return self.reward_coff[0] * hard + self.reward_coff[1] * soft

    #计算矩阵C与b的乘积
    def compute_Cb(self, b):
        return np.dot(self.C, b)

    #计算Cb-x
    def compute_Cbx(self, b):
        return self.compute_Cb(b) - self.x

    #统计得到当前cb-x的模长（这个值采用绝对值加和，如果越小，说明b的效果越好）
    def compute_v(self, observation):
        Cbx = observation
        sum_Cbx = np.linalg.norm(Cbx, 1)
        return sum_Cbx

    #随机一个cl长度的01向量
    def rand_action(self, cl):
        return mc.random_vecb(cl)

    # 将Cb-x作为状态空间
    def generate_state(self, b):
        return self.compute_Cbx(b)
