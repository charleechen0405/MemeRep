# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:20:08 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def selection(P,label,adjData,obj):
    Pparent = []
    lenP = len(P)
    for i in range(0, int(lenP / 2)):
        player1_index = random.randint(0, lenP - 1)
        player2_index = random.randint(0, lenP - 1)
        while (player1_index == player2_index):
            player2_index = random.randint(0, lenP - 1)
        player1 = P[player1_index][:]
        player2 = P[player2_index][:]
        if (obj[player1_index] >= obj[player2_index]):
            Pparent.append(player1)
        else:
            Pparent.append(player2)

    return Pparent,obj