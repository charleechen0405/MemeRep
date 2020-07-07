# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:17:06 2020

@author: Administrator
"""
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time


def localSearch(p,adjData,label,g,Gmax):
    #将每个节点的特征向量向邻居中心节点靠拢，邻居中心节点是综合考虑所有邻居节点，依照每个节点的度在邻居集中所占比重得出的

    for i in range(len(p)):
        indexs = [j for j in range(len(p)) if adjData[i][j]!=0]
        if len(indexs)!=0:
            degree = []
            ptemp = []
            for vertex in indexs:
                degree.append(sum(adjData[vertex]))
                ptemp.append(p[vertex])
            degreeNorm = np.array(degree)/sum(degree)
            c = np.zeros([1,len(p[0])])
            for j in range(len(degree)):
                c += (degreeNorm[j]*ptemp[j])
            p[i] = p[i] + 0.1*(c-p[i])
        else:
            pass
    
    #将每个社区的点向该社区的中心点靠拢，中心点为该社区所有点的特征平均值
    if(g>0.8*Gmax):
        k = max(label)+1
        for i in range(k):
            indexs = []
            community = []
            for j in range(len(p)):
                if(label[j]==i):
                    indexs.append(j)
                    community.append(p[j])
            center = np.sum(community,axis=0)/len(community)        
            for index in indexs:
                p[index] = p[index] + 0.1*(center-p[index])
            
    return p