# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:46:15 2019

@author: Administrator
"""

import numpy as np
import pandas as pd

m = 1000
n = 500
V = np.random.uniform(0,1,(m,n)) + 1

r = 300
W = np.random.uniform(0,1,(m,r)) + 1
H = np.random.uniform(0,1,(r,n)) + 1
H.shape

Min_Value = 10**10
Temp2 = 10**15
while(1):
    Temp2 = F
    X = V/Temp
    W = W * np.dot(X, np.transpose(H))
    W = W / np.sum(W, 0) # Let the sum of every column be 1.
    H = H * np.dot(np.transpose(W), X)
    Temp = np.dot(W, H)
    F = np.sum(np.log(Temp)*V) - np.sum(Temp)
    Diff = F - Temp2
    if(np.abs(np.sum(Diff)) < Min_Value):
        Min_Value = np.abs(np.sum(Diff))
        print(F)
    if(np.abs(np.sum(Diff)) < 10**-3):
        break


