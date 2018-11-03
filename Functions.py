#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import sys
import random
from datetime import datetime

def PrintWithTime(string=None):
    if string is None:
        print("{0}".format(datetime.now().strftime("%m-%d %H:%M:%S")))
    else:
        print("{0}  {1}".format(datetime.now().strftime("%m-%d %H:%M:%S"), string))


def ProgressBar(progress):
    progress *= 100
    progress = round(progress)
    a = '=' * int(progress) + '-' * (100 - int(progress))
    a = a[:47] + (' 100 ' if progress >= 100 else " %2d%% " %
                                                  progress) + a[52:] + '\n'
    print(a)


def BarFormat(string):
    l = len(string)
    a = (100 - 2 - l) // 2
    b = 2 * a + 2 + l == 99
    return '=' * a + ' {0} '.format(string) + '=' * (a + 1) if b else '=' * a + ' {0} '.format(string) + '=' * a

def ObjectFunction(C, b, x):
    # || Cb - x ||
    # [K, ] * [K, D] - [D, ]
    return np.linalg.norm(b.dot(C) - x)


def InitializeBinary(k):
    """Init a random b (0, 1) ^ [K, 1] for first iteration

    Args:
        k (int): the num of b
    """
    b = np.zeros([k, ], dtype=np.int8)

    """ Primal test, half of b set to 1, and another half is 0 """
    choice = np.random.choice(k, k // 2)
    b[choice] = 1
    return b


def OptimizeC(C, X, B, learning_rate=1e-2):
    """[summary]
    
    Args:
        C (ndarray): Codebook, [K, D]
        X (ndarray): Batch of X, [N, D]
        B (ndarray): Code of this batch, [N, K]
    """
    delta = 2 * B.T.dot(B.dot(C) - X)
    return C - learning_rate * delta



def MiniBatchOptimizeC(C, X, B, learning_rate=1e-2):
    """[summary]
    
    Args:
        C (ndarray): Codebook, [K, D]
        X (ndarray): Batch of X, [N, D]
        B (ndarray): Code of this batch, [N, K]
    """
    N = X.shape[0]
    delta = 2 * B.T.dot(B.dot(C) - X)
    return C - learning_rate * delta / N