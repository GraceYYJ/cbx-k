# encoding: utf-8
import numpy as np
import random_c as rc
import my_common as mc
import gradient_c as gc
import ddpg_b as db
import math
import torch
import fvecs_read
from tabusearch import *

#问题综述：找到X的最佳量化模式，数学表达为求解CB-X的最小值，C和B迭代进行，C为根据维度的随机初始矩阵
#X是数据组成的矩阵，维度为d*n
#C是codebook，是一个d*k的矩阵
#B是编码矩阵,并且元素全为0和1，维度k*n

#解决思路：
# 1、在使用禁忌搜索的方法迭代z(z=5)轮之后，得到计算过程中相对靠谱的C
# 2、使用强化学习方案，随机选取X中的任意t(t=5)个向量作为学习样本,使用较大的epoach次数与step次数，完成t个样本的最佳b路径选择模型
# 3、对剩余的n-t个样本，依次对比t个样本，选取一范式差值最小的t所在的模型，进行finetune运算，得到各自的b值
# 4、整理n个b值形成B后，使用梯度下降（矩阵求导）的方式得到新的C
# 5、返回2进行一下轮迭代，直到迭代次数终止
# 6、收集最后一次t个样本及模型，用于可能到来的查询

#参数设定
static_d = 0
static_n = 0
static_k = 16
org_X = []
static_X = []
X_loss = 0
_C = []
_B = []
seed_c = 4
X_loss_thr = 88
#禁忌搜索的最大迭代次数
tabu_t = 1 #5
#_C与_B的循环的最大迭代次数
Max_iter_num = 18
#输出的b的维度
action_output_num = 16
coff = [0.8, 0.2]
gamma = 0.99
tau = 0.001
hidden_size = 128

#读取所有的X
# org_X = np.loadtxt(open("mock_data", "rb"), delimiter=" ", skiprows=0)
org_X = fvecs_read.fvecs_read("./datasets/siftsmall/siftsmall_base.fvecs")
# print(org_X.shape)
static_X = org_X.T
static_n = org_X.shape[0]
static_d = static_X.shape[0]

#在禁忌搜索z轮迭代下进行强化学习的_C和_B得到
#进入强化学习的迭代循环
temp_iter_num = tabu_t
#从X中随机得到t个样本(以多进程的方式对他们进行并行的模型训练)

if __name__ == "__main__":
    # 使用禁忌搜索得到_C和_B
    _C, _B = tabuSearch(tabu_t)
    i = 0
    while True:
        #对抽取的t个样本中的第一个样本进行学习
        # _C是当前codebook，b是这个样本对应的码值(来源于上一阶段的禁忌搜索)，x是抽取到的样本
        # model_ddpg1 = db.DDPGB(_C, b, x, action_output_num)
        model_ddpg1 = db.DDPGB(_C, _B, org_X[i:i+1].T, action_output_num)
        float_b1 = model_ddpg1.generate_B(coff, gamma, tau, hidden_size, static_d)
        b1 = mc.binazation(float_b1)
        # 对于剩余的x样本，比对和这t个样本的欧式距离，选取距离最小的进行模型读取和finetune，得到各自的b
        # 组合这些b形成_B
        # 根据得到的B和X通过矩阵求导和梯度下降法得到新的C
        obj_gc = gc.GrandientC(_C, _B, static_X)
        _C, _C_loss = obj_gc.degrade(0.008, math.pow(10, -9))

        #检验结果
        X_loss = mc.matrix_distance(np.dot(_C, _B) - static_X)
        print("the {} epoach loss is {}".format(temp_iter_num, X_loss))

        temp_iter_num += 1
        if X_loss < X_loss_thr:
            break
        if temp_iter_num > Max_iter_num:
            break
        i+=1

    #给出最后的C和B
    print("_C:",_C)

    print("_B:",_B)
