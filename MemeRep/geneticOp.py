# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:20:37 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def geneticOp(Pparent, pc, pm ,mu, mum):
    Pchild = []
    max_var = 1
    min_var = 0
    for i in range(0, len(Pparent), 2):
        x1 = Pparent[i][:]
        x2 = Pparent[i + 1][:]
        x1_new = np.zeros(x1.shape)
        x2_new = np.zeros(x2.shape)
        # 交叉
        if (random.random() < pc):
            bq = np.zeros([len(x1),len(x1[0])])
            for j in range(len(x1)):
                for k in range(len(x1[0])):
                    u = random.random()
                    if u<=0.5:
                        bq[j][k] = ((2.0*u)**(1.0/(mu+1)))
                    else:
                        bq[j][k] = ((1.0/(2.0*(1-u)))**(1.0/(mu+1)))
            for j in range(len(x1[0])):
                x1_new[:,j] = 0.5*((1+bq[:,j])*x1[:,j]+(1-bq[:,j])*x2[:,j])

                x2_new[:,j] = 0.5*((1-bq[:,j])*x1[:,j]+(1+bq[:,j])*x2[:,j])

            for j in range(len(x1)):
                for k in range(len(x1[0])):
                    if(x1_new[j][k] > max_var):
                        x1_new[j][k] = max_var
                    elif(x1_new[j][k] < min_var):
                        x1_new[j][k] = min_var
                    if (x2_new[j][k] > max_var):
                        x2_new[j][k] = max_var
                    elif(x2_new[j][k] < min_var):
                        x2_new[j][k] = min_var
            x1 = x1_new
            x2 = x2_new
        #变异
        for j in range(len(x1)):
            for k in range(len(x1[0])):
                if(random.random()<pm):
                    r = random.random()
                    if r<0.5:
                        delta = (2*r)**(1/(mum+1)) -1
                    else:
                        delta = 1-(2*(1-r))**(1/(mum+1))
                    x1[j][k] = x1[j][k] + delta
                    x2[j][k] = x2[j][k] - delta
                    if(x1[j][k]>max_var):
                        x1[j][k] = max_var
                    elif(x1[j][k]<min_var):
                        x1[j][k] = min_var
                    if(x2[j][k]>max_var):
                        x2[j][k] = max_var
                    elif(x2[j][k]<min_var):
                        x2[j][k] = min_var
        Pchild.append(x1)
        Pchild.append(x2)
    return Pchild
