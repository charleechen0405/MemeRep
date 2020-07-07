

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 04:24:57 2017

@author: charleechen
"""

import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
import time
from InitialPop import InitialPop
from clustering import clustering
from cal_fit import cal_fit
from selection import selection
from geneticOp import geneticOp
from localSearch import localSearch
from updatePop import updatePop




if __name__ == "__main__":
    
    #读取edgelist文件，生成邻接矩阵，节点标号从1开始
    start = time.time()
    edges_raw = open(r'E:\github\MemeRep\docs\Edgekarate.txt')
    edges = []
    for line in edges_raw.readlines():
        line = line.split()
        edges.append([int(line[0]),int(line[1])])
    edges = np.array(edges)
    nodes = max(max(edges[:,0]),max(edges[:,1]))
    adjData = np.zeros([nodes,nodes])
    for i in range(len(edges)):
        adjData[edges[i][0]-1][edges[i][1]-1] = 1
        adjData[edges[i][1]-1][edges[i][0]-1] = 1
        

    vertex_num = len(adjData)
    feature_dim = 180 #表示维度
    pc = 0.9          #交叉概率
    pm = 0.1          #变异概率
    pop = 100         #种群大小
    Gmax = 50         #迭代次数
    k = 4             #指定类别数
    obj = np.zeros([pop])
    P = InitialPop(pop, vertex_num, feature_dim, adjData) #仿照meme算法有启发的初始化特征矩阵
    label_parent = clustering(P,k) #kmeans聚类确定标签
    for i in range(len(P)):
        obj[i] = cal_fit(label_parent[i], adjData)
    g = 0

    while (g<Gmax):
        if (g < 0.5*Gmax):
            mu = 1
        else:
            mu = 20
        mum = 20
        Pparent,obj_parent = selection(P,label_parent,adjData,obj)
        Pchild = geneticOp(Pparent,pc,pm,mu,mum)
        
        label_child = clustering(Pchild,k)
        obj_child = np.zeros([len(Pchild)])
        for i in range(len(Pchild)):
            obj_child[i] = cal_fit(label_child[i],adjData)
            
        index_best_child = list(obj_child).index(max(list(obj_child)))
        Pchild[index_best_child] = localSearch(Pchild[index_best_child],adjData,label_child[index_best_child],g,Gmax)
        
        m1 = KMeans(n_clusters = k)
        m1.fit(Pchild[index_best_child])
        label_temp = m1.labels_.tolist()
        obj_child[index_best_child] = cal_fit(label_temp,adjData)
        label_child[index_best_child] = label_temp
        
        P,label_parent,obj = updatePop(P,Pchild,obj_parent,obj_child,label_parent,label_child)
        g = g+1
        print('当前迭代次数：'+ str(g))
        
    index_best = list(obj).index(max(list(obj)))
    best = P[index_best][:]
    np.savetxt(r'E:\github\MemeRep\output\karate.csv',best,delimiter=',')
    print('complete')

    
    


  
  
    

