# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:19:37 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def InitialPop(pop,vertex_num,feature_dim,adjData):
    P = []
    alpha=0.2
    eta = 0.7
    for i in range(pop):
        temp1=[]
        t=0
        for j in range(vertex_num):
            temp2 = np.random.rand(feature_dim)
            temp1.append(temp2)

        while(t<alpha*vertex_num):
            xi = random.randint(1, len(adjData))
            for j in range(vertex_num):
                if(adjData[xi-1][j]==1):
                    temp1[j] = temp1[j]+eta*(temp1[xi-1]-temp1[j])
            t = t+1

        P.append(temp1)

    return np.array(P)