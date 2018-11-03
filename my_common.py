# encoding: utf-8
import numpy as np
import math
import random

#根据给定的范围随机选取规定个数的数字
def random_select(left, right, candidate):
    return random.sample(range(left, right), candidate)

#根据给定范围随机将部分值的符号改变
def random_change_sgn(left, right, candidate, vec):
    list = random_select(left, right, candidate)
    for i in list:
        vec[i] = -vec[i]
    return vec

#根据大于0则为1，其他为0为0的原则对向量进行二值化
def binazation(vec):
    return np.int64(vec > 0.5)

#随机一个长度为len的01向量
def random_vecb(len):
    return np.array([i % 2 for i in np.random.randint(1, 100, size=(len))])

def random_matB(row,col):
    matB = []
    for i in range(col):
        matB = np.hstack((matB, random_vecb(row)))
    return np.array(matB).reshape(col, row).T

#计算两个矩阵间的距离
def matrix_distance(mat, norm=1):
    if norm == 1:
        fmat = np.fabs(mat)
        return np.sum(fmat) / mat.size
    else:
        fmat = np.multiply(mat, mat)
        return np.sqrt(np.sum(fmat))/mat.size

#输入的x为标准矩阵，处理时需要转置
def trans_intmat_binmat(x, hd):
    ture_x = x.T
    row = ture_x.shape[0]
    col = ture_x.shape[1]
    binmat = []

    for i in range(row):
        binvec = trans_int10_vector2(ture_x[i], hd)
        binmat = np.hstack((binmat, binvec))

    return np.array(binmat).reshape(row, col*hd).T

def trans_b10_vector2(int_b,hd, is_v=True):
    #math.floor(x)向下取整
    #横向并联两个矩阵或向量np.hstack((a,b))
    b_vector = np.zeros(0)
    int_b = int_b.squeeze()
    for num in int_b:
        b_vector = np.hstack((b_vector, trans_int10_vector2(math.floor(num),hd)))
    if is_v:
        return b_vector.T
    else:
        return b_vector

def trans_int10_vector2(x, hd):
    num = x
    bin_vector = np.zeros(hd)
    while True:
        hd = hd - 1
        num, remainder = divmod(num, 2)
        if remainder == 1:
            bin_vector[hd] = 1
        if hd == 0:
            return bin_vector

#输入的x为标准矩阵，处理时需要先转置
def trans_binmat_intmat(x, hd):
    ture_x = x.T
    row = ture_x.shape[0]
    col = ture_x.shape[1]
    intmat = []

    for i in range(row):
        intvec = trans_binvec_intvec(ture_x[i], hd)
        intmat = np.hstack((intmat, intvec))

    return np.array(intmat).reshape(row, col//hd).T

def trans_binvec_intvec(x, hd):
    len = x.shape[0]
    mul = 0
    intvec = np.zeros(len//hd)
    while True:
        vec = x[hd * mul:hd * (mul+1)]
        intvec[mul] = trans_binvec_int(vec)
        mul = mul + 1
        if hd * mul == len:
            return intvec

def trans_binvec_int(x):
    len = x.shape[0]
    num = 0
    for i in range(len):
        num = num + x[i] * math.pow(2, len - i - 1)
    return num

def up2dim(x):
    x = np.expand_dims(x, axis=2)
    x = np.expand_dims(x, axis=3)
    return x









