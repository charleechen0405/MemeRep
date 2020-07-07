# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:18:25 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def cal_fit(x1, adjData):
    lamda = 0.5
    x = x1[0:len(x1)]
    k = len(list(set(x)))
    label = list(set(x))

    KKM = 0
    RC = 0
    for i in range(k):

        numi = list(x).count(label[i])
        temp = [index1 for index1 in range(len(x)) if x[index1] == label[i]]  # 找到类别为label[i]的下标
        all_com = list(itertools.combinations(temp, 2))  # 找出该社团内所有点的下标再进行组合
        L1 = 0
        for index2 in all_com:
            if (adjData[index2[0]][index2[1]] == 1):
                L1 = L1 + 1
        degree_i = 0
        for j in temp:
            degree_i = degree_i + sum(adjData[j])

        L2 = degree_i - 2 * L1
        RC = RC + L2 / numi
        KKM = KKM - L1 / numi
    fit = -(2 * lamda * KKM + 2 * (1 - lamda) * RC)
    return fit