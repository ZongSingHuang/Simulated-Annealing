# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:07:21 2021

@author: zongsing.huang
"""

import itertools

import numpy as np

#%% 題庫
benchmark = np.array([[ 0, 19, 92, 29, 49, 78,  6],
                      [19,  0, 21, 85, 45, 16, 26],
                      [92, 21,  0, 24, 26, 87, 47],
                      [29, 85, 24,  0, 76, 17,  8],
                      [49, 45, 26, 76,  0, 90, 27],
                      [78, 16, 87, 17, 90,  0, 55],
                      [ 6, 26, 47,  8, 27, 55,  0]])

#%% 函數定義
def fitness(X, benchmark):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    D = X.shape[1]
    F = np.zeros(P)
    
    for i in range(P):
        X_new = np.append(X[i], X[i, 0])
        
        for j in range(D):
            st = X_new[j].astype(int)
            ed = X_new[j+1].astype(int)
            F[i] += benchmark[st, ed]
    
    return F

def swap(X):
    D = X.shape[0]
    idx = np.arange(D)
    comb = list(itertools.combinations(idx, 2))
    X_new = np.zeros([len(comb), D])
    
    for i, (j, k) in enumerate(comb):
        X_new[i] = X.copy()
        X_new[i, j], X_new[i, k] = X_new[i, k], X_new[i, j]
    
    return X_new

#%% 參數設定
D = benchmark.shape[1] # 維度
T = 300 # 初始溫度
Tf = 0.1 # 終止溫度
beta = 0.97 # 衰退率

#%% 初始化
X = np.random.choice(D, size=D, replace=False) # 初始解
F = fitness(X, benchmark) # 初始適應值

while T>Tf:
    # 更新
    X_set = swap(X)
    F_set = fitness(X_set, benchmark)
    idx = F_set.argmin()
    X_new = X_set[idx]
    F_new = F_set[idx]
    
    # 計算新舊適應值差距
    delta = F_new - F
    
    # 新的解比較差
    if delta>0:
        r = np.random.uniform()
        
        # 一定機率接受較差的解
        if r<=np.minimum(1, np.exp(-delta/T)):
            X = X_new.copy()
            F = F_new.copy()
    
    # 新的解比較好，直接接受
    else:
        X = X_new.copy()
        F = F_new.copy()
    
    # 更新溫度
    T = beta * T