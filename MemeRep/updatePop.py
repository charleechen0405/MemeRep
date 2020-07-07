# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:18:48 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def updatePop(P1,P2,obj1,obj2,label_parent,label_child):
    P = np.vstack((P1,P2))
    label = label_parent + label_child
    obj = np.vstack((obj1.reshape([len(obj1),1]),obj2.reshape([len(obj2),1])))
    P_temp = list(zip(P,label,obj))
    Pnew = np.array(sorted(P_temp,key = lambda x:x[-1],reverse = True))

    return Pnew[0:len(P1),0].tolist(),Pnew[0:len(P1),1].tolist(),Pnew[0:len(P1),2]