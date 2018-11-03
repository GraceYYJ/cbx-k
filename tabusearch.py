#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
# from sift_read import sift10k_read
from ITS import ITS_Instance
from Functions import *
from multiprocessing import Pool
from time import perf_counter

# 在这里用的是BC-X，我们用的是CB-X
# K = 256
K = 16
# D = 64
D = 128
batch_size = 1
Total = batch_size * 1
jump_count = 4

Codes = np.zeros([Total, K], dtype=np.int)

# [k, D]
C = np.random.rand(K, D)

b = np.random.rand(Total, K)

bits = np.zeros([Total, K], dtype=np.int)

bits[b > 0.5] = 1

X = bits.dot(C)

X = X + np.random.randn(Total, D)


def partition(arg):
    index, Query = arg
    distortion = 0
    N = Query.shape[0]
    codes = np.zeros([N, K], dtype=np.int)
    i = 0
    for q in Query:
        tabu = ITS_Instance(5 * D, jump_count, C, q)
        tabu.run()
        codes[i] = tabu.BestSolution
        d = tabu._fit(tabu.BestSolution)
        distortion += d
        i += 1
        # print((index, i))
    print(index, 'Completed')
    return distortion, codes


learning_rate = 1e-2

curve = []

def tabuSearch(tabu_t):
    global C,learning_rate,curve
    for _ in range(tabu_t):
        distortion = 0

        np.random.shuffle(X) #将X按行重新排列

        splitted = np.split(X, Total // batch_size) ##将X分成Total//batch_size个组

        l = list()
        i = 0
        for s in splitted:
            l.append((i, s))
            i += 1
        with Pool(4) as p:
            results = p.map(partition, l) #l为list，对l中每个元素进行partition运算，一次运算四个元素，因为只有4个进程
            k = 0
            i = 0
            for res in results:
                dis, code = res
                n = code.shape[0]
                distortion += dis
                Codes[k:k+n] = code #按组更新
                C = MiniBatchOptimizeC(C, splitted[i], code, learning_rate)
                i += 1
                k += n
        distortion = distortion / X.shape[0]
        PrintWithTime(distortion)
        curve += distortion
        learning_rate *= 0.99
    # print("C:", C)
    # print("C.shape:", C.shape)
    # print("Codes:", Codes)
    # print("Codes.shape:", Codes.shape)
    # print("curve:", curve)
    # print("X.shape:",X.shape)
    # curve = np.array(curve)
    # print("np.array(curve):", np.array(curve))
    # exit()
    # np.save('./distortion_curve', curve)
    return C.T,Codes.T
