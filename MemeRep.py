

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 04:24:57 2017

@author: charleechen
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import itertools
from math import log
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

def updatePop(P1,P2,obj1,obj2,label_parent,label_child):
    P = np.vstack((P1,P2))
    label = label_parent + label_child
    obj = np.vstack((obj1.reshape([len(obj1),1]),obj2.reshape([len(obj2),1])))
    P_temp = list(zip(P,label,obj))
    Pnew = np.array(sorted(P_temp,key = lambda x:x[-1],reverse = True))

    return Pnew[0:len(P1),0].tolist(),Pnew[0:len(P1),1].tolist(),Pnew[0:len(P1),2]

def clustering(P,k):
    labelMatrix = []
    for i in range(len(P)):
        m1 = KMeans(n_clusters=k)
        m1.fit(P[i])
        label = m1.labels_.tolist()
        labelMatrix.append(label)

    return labelMatrix

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


if __name__ == "__main__":
    
    start = time.time()
    edges_raw = open('/home/amax/chencheng/lingshou/edgeList.txt')
    edges = []
    for line in edges_raw.readlines():
        line = line.split()
        edges.append([int(line[0]),int(line[1])])
    edges = np.array(edges)
    nodes = max(max(edges[:,0]),max(edges[:,1]))+1
    adjData = np.zeros([nodes,nodes])
    for i in range(len(edges)):
        adjData[edges[i][0]][edges[i][1]] = 1
        adjData[edges[i][1]][edges[i][0]] = 1
        

    vertex_num = len(adjData)
    feature_dim = 180
    pc = 0.9
    pm = 0.1
    pop = 100
    Gmax = 50
    k = 4
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
        print('当前迭代次数：'+ g)
        
    index_best = list(obj).index(max(list(obj)))
    best = P[index_best][:]
    np.savetxt('/home/amax/chencheng/cora_1_1.csv',best,delimiter=',')
    print('complete')

    
    


  
  
    

