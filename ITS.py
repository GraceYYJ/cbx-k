#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import sys
import random
from time import perf_counter


class ITS(object):
    def __init__(self, maxIteration: int, maxRunningTicks: int, initFunc, fitFunc):
        """禁忌搜索初始化

        Args:
            maxIteration (int): tabu search 迭代次数
            maxRunningTicks (int): 停机时间, 秒
            initFunc ([type]): 解的初始化函数
            fitFunc ([type]): 目标评价函数
        """

        self.BestSolution = None
        self.BestFitness = 0.0

        self._initSolution = initFunc
        self._fit = fitFunc

        self._maxIteration = maxIteration
        self._maxRunningTicks = maxRunningTicks

    def run(self):

        startTime = perf_counter()

        currentSolution = self._initSolution()

        D = currentSolution.shape[0]
        oneThird = D // 3

        currentFitness = self._fit(currentSolution)

        tabuList = np.zeros([D], dtype=np.float)

        self.BestFitness = currentFitness
        self.BestSolution = currentSolution

        # 非禁忌候选集中的解
        nextSolution = None
        nextFitness = int(sys.maxsize)
        nextTabu0 = None
        fitList = np.zeros([D], dtype=np.float)

        # 禁忌候选集中的解
        tabuSolution = None
        netxtTabuFitness = int(sys.maxsize)
        nextTabu1 = None
        tabufitList = np.zeros([D], dtype=np.float)

        total_iter = 0

        while True:
            bestList = None
            tabuList = np.zeros([D], dtype=np.float)
            # 迭代限制
            for i in range(self._maxIteration):
                nextFitness = int(sys.maxsize)
                netxtTabuFitness = int(sys.maxsize)

                success = False

                # 邻域候选集中查找
                for j in range(D):
                    neighbor = np.copy(currentSolution)
                    neighbor[j] = 1 - neighbor[j]
                    val = self._fit(neighbor)
                    # 是否满足禁忌要求
                    if i > tabuList[j]:
                        fitList[j] = val
                        # 选择邻域中评价最好的解作为候选
                        if nextFitness > val:
                            nextSolution = neighbor
                            nextFitness = val
                            # 记录位置
                            nextTabu0 = j
                            success = True
                    else:
                        tabufitList[j] = val
                        # 选择邻域中评价最好的解作为候选
                        if netxtTabuFitness > val:
                            tabuSolution = neighbor
                            netxtTabuFitness = val
                            # 记录位置
                            nextTabu1 = j

                # 赦免：禁忌表已满，或禁忌中的值比最好值还好
                if not success or (nextFitness > netxtTabuFitness and netxtTabuFitness < self.BestFitness):
                    # 更新禁忌表，当前值
                    # tabuList[nextTabu1] = i + (D / 150) + random.randint(1, 10)
                    tabuList = np.zeros([D], dtype=np.float)
                    currentSolution = tabuSolution
                    currentFitness = netxtTabuFitness
                    fitnessList = tabufitList
                else:
                    # 更新禁忌表，当前值
                    tabuList[nextTabu0] = i + (D / 150) + random.randint(1, 10)
                    currentSolution = nextSolution
                    currentFitness = nextFitness
                    fitnessList = fitList

                # print(currentFitness)
                # 更新最佳值
                if currentFitness < self.BestFitness:
                    self.BestSolution = currentSolution
                    self.BestFitness = currentFitness
                    bestList = fitnessList
                    # print('best updated: ')
                    # print(self.BestSolution)
                    # print(self.BestFitness)

            # 停机
            if total_iter > self._maxRunningTicks:
                return

            total_iter += 1

            # 跳坑
            if bestList is not None:
                bestBits = np.argpartition(-1 *
                                           bestList, kth=oneThird)[:oneThird]
                # print('flip:', bestBits)
                # print('from', currentSolution.reshape([1, -1]), 'jump to:')
                currentSolution[bestBits] = 1 - currentSolution[bestBits]
                # print(currentSolution.reshape([1, -1]))
                currentFitness = self._fit(currentSolution)
                # print(str(self.BestFitness) + ' -> ' + str(currentFitness))
                tabuList[:] = 0
                tabuList[bestBits] = (D / 150) + random.randint(1, 10)


class ITS_Instance():
    def __init__(self, maxIteration: int, maxRunningTicks: int, C, query):
        """禁忌搜索初始化

        Args:
            maxIteration (int): tabu search 迭代次数
            maxRunningTicks (int): 停机时间, 秒
        """

        self.BestSolution = None
        self.BestFitness = 0.0

        self._maxIteration = maxIteration
        self._maxRunningTicks = maxRunningTicks

        self._C = C
        self._x = query

        self._k = C.shape[0]

    def _fit(self, b):
        # faster than matrix operation, see test_norm.py
        return np.linalg.norm(b.dot(self._C) - self._x)

    def _matfit(self, b):
        # norm for each row, [N, K] * [K, D] - [N, D]
        return np.linalg.norm(b.dot(self._C) - self._x, axis=1)

    def _initSolution(self):
        b = np.zeros([self._k, ], dtype=np.int8)
        """ Primal test, half of b set to 1, and another half is 0 """
        choice = np.random.choice(self._k, self._k // 2)
        b[choice] = 1
        return b

    def _updateFitnessList(self, solution):
        # get the fitness list that using this solution and flip each bit
        D = solution.shape[0]
        # [D, D]
        neighbors = np.tile(solution, [D, 1])
        # flip bits for every row
        neighbors[range(D), range(D)] = 1 - neighbors[range(D), range(D)]

        # compute fit function for each row
        return self._matfit(neighbors)

    def run(self):

        currentSolution = self._initSolution()

        D = currentSolution.shape[0]
        oneThird = D // 4

        currentFitness = self._fit(currentSolution)

        tabuList = np.zeros([D], dtype=np.float)

        self.BestFitness = currentFitness
        self.BestSolution = currentSolution

        # 非禁忌候选集中的解
        nextSolution = None
        nextFitness = int(sys.maxsize)
        nextTabu0 = None

        # 禁忌候选集中的解
        tabuSolution = None
        netxtTabuFitness = int(sys.maxsize)
        nextTabu1 = None

        total_iter = 0

        while True:
            bestList = None
            tabuList = np.zeros([D], dtype=np.float)
            # 迭代限制
            for i in range(self._maxIteration):
                nextFitness = int(sys.maxsize)
                netxtTabuFitness = int(sys.maxsize)

                success = False

                # 邻域候选集中查找

                # [D, D]
                neighbors = np.tile(currentSolution, [D, 1])
                # flip bits for every row
                neighbors[range(D), range(D)] = 1 - neighbors[range(D), range(D)]

                # compute fit function for each row
                vals = self._matfit(neighbors)

                for j in range(D):
                    # 是否满足禁忌要求
                    if i > tabuList[j]:
                        # 选择邻域中评价最好的解作为候选
                        if nextFitness > vals[j]:
                            nextSolution = neighbors[j]
                            nextFitness = vals[j]
                            # 记录位置
                            nextTabu0 = j
                            success = True
                    else:
                        # 选择邻域中评价最好的解作为候选
                        if netxtTabuFitness > vals[j]:
                            tabuSolution = neighbors[j]
                            netxtTabuFitness = vals[j]
                            # 记录位置
                            nextTabu1 = j

                # 赦免：禁忌表已满，或禁忌中的值比最好值还好
                if not success or (nextFitness > netxtTabuFitness and netxtTabuFitness < self.BestFitness):
                    # 更新禁忌表，当前值
                    # tabuList[nextTabu1] = i + (D / 150) + random.randint(1, 10)
                    tabuList[:] = 0
                    tabuList[nextTabu1] = i + (D / 150) + random.randint(1, 10)
                    currentSolution = tabuSolution
                    currentFitness = netxtTabuFitness
                else:
                    # 更新禁忌表，当前值
                    tabuList[nextTabu0] = i + (D / 150) + random.randint(1, 10)
                    currentSolution = nextSolution
                    currentFitness = nextFitness

                # print(currentFitness)
                # 更新最佳值
                if currentFitness < self.BestFitness:
                    self.BestSolution = currentSolution
                    self.BestFitness = currentFitness
                    bestList = self._updateFitnessList(self.BestSolution)
                    # print('best updated: ')
                    # print(self.BestSolution)
                    # print(self.BestFitness)

            # 停机
            if total_iter > self._maxRunningTicks:
                return

            total_iter += 1

            # 跳坑
            if bestList is not None:
                bestBits = np.argpartition(bestList, kth=oneThird)[:oneThird]
                # print('flip:', bestBits)
                # print('from', currentSolution.reshape([1, -1]), 'jump to:')
                currentSolution[bestBits] = 1 - currentSolution[bestBits]
                # print(currentSolution.reshape([1, -1]))
                currentFitness = self._fit(currentSolution)
                # print(str(self.BestFitness) + ' -> ' + str(currentFitness))
                tabuList[:] = 0
                tabuList[bestBits] = (D / 150) + random.randint(1, 10)
