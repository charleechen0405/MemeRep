# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:19:15 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def clustering(P,k):
    labelMatrix = []
    for i in range(len(P)):
        m1 = KMeans(n_clusters=k)
        m1.fit(P[i])
        label = m1.labels_.tolist()
        labelMatrix.append(label)

    return labelMatrix