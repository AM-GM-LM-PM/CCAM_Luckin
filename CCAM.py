# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:11:34 2019

User Behavior Analysis and Commodity Recommendation for Point-Earning Apps
Yu-Ching Chen, Chia-Ching Yang, Yan-Jian Liau, Chia-Hui Chang
National Central University
Taoyuan, Taiwan
TAAI2016

@author: Administrator
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib
import numba
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@numba.jit
def F_New_to_F_hat(F_New, U_group, A_group, Row_number_F, Col_number_F):
    F_hat = np.random.randn(Row_number_F, Col_number_F)
    for i in range(Row_number_F):
        for j in range(Col_number_F):
            F_hat[i,j] = F_New[U_group[i], A_group[j]] # No need to -1
    return F_hat

@numba.jit
def H_New_tilde_to_H_hat(H_New_tilde, U_group, Row_number_F, Col_number_H):
    H_hat = np.random.randn(Row_number_F, Col_number_H)
    for i in range(Row_number_F):
        for j in range(Col_number_H):
            H_hat[i,j] = H_New_tilde[U_group[i], j] # No need to -1
    return H_hat
        
@numba.jit
def G_New_tilde_to_G_hat(G_New_tilde, A_group, Col_number_F, Col_number_G):
    G_hat = np.random.randn(Col_number_F, Col_number_G)
    for i in range(Col_number_F):
            for j in range(Col_number_G):
                G_hat[i, j] = G_New_tilde[A_group[i], j]
    return G_hat
        
@numba.jit
def RemoveZero(X):
    [m,n] = X.shape
    for i in range(m):
        for j in range(n):
            if(abs(X[i,j]) <= 10**-11):
                X[i,j] = 10**-11
    return X

@numba.jit
def JudgeZero(X):
    [m,n] = X.shape
    for i in range(m):
        for j in range(n):
            if(X[i,j] < 0):
                X[i,j] = -X[i,j]
    return X


def CCAM():
    varphi = 5000
    Lambda = 0.4

    User_Profile = pd.read_csv("User_Profile_subset2.csv", encoding = 'utf-8')
    User_Profile = User_Profile.drop('Unnamed: 0', axis = 1)
    Phone_No_Index = User_Profile['phone_no'].values
    User_Profile = User_Profile.drop('phone_no', axis = 1)

    H = User_Profile.values[1:,:]
    H = JudgeZero(H)
    # NOTE: pd.DataFrame(H).values == H

    User_Item = pd.read_csv("User_Item_subset2.csv", encoding = 'utf-8')
    User_Item = User_Item.drop(User_Item.index[0])
    # Remove the first row and the header, irrelavant info
    # User_Item['commodity_code'] = User_Item['commodity_code'].astype(np.int64)
    User_Item = User_Item.loc[User_Item['phone_no'].isin(Phone_No_Index)]

    F = User_Item.values[:,1:]

    Item_Profile = pd.read_csv("Item_Profile_subset2.csv", encoding = 'utf-8', skiprows = 1)
    Item_Profile = Item_Profile.drop(Item_Profile.index[0])
    INDEX = User_Item.columns[1:]
    Item_Profile = Item_Profile.loc[Item_Profile["commodity_code"].isin(INDEX)]
    G = Item_Profile.values[:,1:]
    # One of the row/column is missing, needs to do another filtering
    
    F = F / np.sum(F)
    G = G / np.sum(G)
    H = H / np.sum(H)
    #print(H[H<0])
    # Normalization
    F = np.asarray(F).astype(np.float64)
    G = np.asarray(G).astype(np.float64)
    H = np.asarray(H).astype(np.float64)
  #  print(H[H<0])
    
    Row_number_F = F.shape[0]
    Col_number_F = F.shape[1]
    Col_number_G = G.shape[1]
    Col_number_H = H.shape[1]
    
    '''
    U_group = np.random.random_integers(1, 9, Row_number_F) - 1
    A_group = np.random.random_integers(1, 9, Col_number_F) - 1
    Wrong! Example:
    [0 4 6 6 3 2 0 7 8]
    The index 8 should be corresponded to the 6th category 
    for number 1, 5 are missing, so the shape
    of conditional prob is 7 not 9, which leads to keyerror.  
    '''
    U_group = np.array([i for i in range(9)]+list(np.random.random_integers(1, 9, Row_number_F - 9) - 1))
    A_group = np.array([i for i in range(3)]+list(np.random.random_integers(1, 3, Col_number_F - 3) - 1))
    
    '''
    This is a toy example.
    varphi = 0.5
    Lambda = 0.4
    U_group = np.array([1, 1, 2, 2, 3, 3]) - 1
    A_group = np.array([2, 1, 2, 1]) - 1
    
    F = np.array([0.05, 0.05, 0.15, 0, 
                  0.05, 0.05, 0.15, 0,
                  0, 0, 0, 0.15,
                  0, 0.05, 0, 0.05,
                  0.05, 0, 0, 0.05,
                  0.05, 0.05, 0.05, 0]).reshape(6, 4) # f(U,A) Toy example, Original Data
    
    F_Boundary_U = np.sum(F, 1) # f(U)
    F_cond = np.transpose(np.transpose(F)/F_Boundary_U) # f(A|U)
    
    F = pd.DataFrame(F)
    F_cond = pd.DataFrame(F_cond)
    B = F.groupby(U_group).sum()
    F_New = B.transpose().groupby(A_group).sum().transpose() # f(U_tilde,A_tilde)
    
    H = np.array([0.05, 0,
                  0.12, 0.05,
                  0, 0.15,
                  0.2, 0.04,
                  0.04, 0.2,
                  0.05, 0.1]).reshape(6, 2) # h(U, L)
    
    H_Boundary_U = np.sum(H, 1) # h(U)
    H_cond = np.transpose(np.transpose(H)/H_Boundary_U) # h(L|U)
    
    H = pd.DataFrame(H)
    H_cond = pd.DataFrame(H_cond)
    H_New_tilde = H.groupby(U_group).sum() # h(U_tilde, L)
    H_tilde_cond = np.transpose(np.transpose(H_New_tilde)/np.sum(H_New_tilde,1)) # h(L|U_tilde)
    
    F_tilde = np.random.randn(6, 4)
    F_tilde = pd.DataFrame(F_tilde) # f_tilde(U, A)

    G = np.array([0.25, 0,
                  0.12, 0.05,
                  0, 0.15,
                  0.2, 0.23]).reshape(4,2)

    G_Boundary_A = np.sum(G, 1) # G(A)
    G_cond = np.transpose(np.transpose(G)/G_Boundary_A) # G(L|A)
    
    G = pd.DataFrame(G)
    G_cond = pd.DataFrame(G_cond)
    G_New_tilde = G.groupby(A_group).sum() # G(A_tilde, L)
    G_tilde_cond = np.transpose(np.transpose(G_New_tilde)/np.sum(G_New_tilde,1)) # h(L|U_tilde)
    
    Row_number_F = 6
    Col_number_F = 4
    Col_number_H = 2
    Col_number_G = 2
    '''
    Q = 1<<25
    flag = 0 # Prevent F from being computed repeatedly
    flag2 = 0 # Prevent F,G,H's 0 from being changed repeatedly
    Num = 1 # Count the number of iteration
    while True:
        print(Num)
        F_Boundary_A = np.sum(F, 0)
        F_Boundary_U = np.sum(F, 1) # f(U)
        F_cond = np.transpose(np.transpose(F)/F_Boundary_U) # f(A|U)

        F = pd.DataFrame(F)
        F_cond = pd.DataFrame(F_cond)
        B = F.groupby(U_group).sum()
        F_New = B.transpose().groupby(A_group).sum().transpose() # f(U_tilde,A_tilde)
        # This operation requires pandas, so need to reverse to numpy
        F = np.array(F).astype(np.float64)
        F_cond = np.array(F_cond).astype(np.float64)
        F_New = np.array(F_New).astype(np.float64)
        
    ### G???
        G_Boundary_A = np.sum(G, 1) # G(A)
        G_cond = np.transpose(np.transpose(G)/G_Boundary_A) # G(L|A)
        
        G = pd.DataFrame(G)
        G_cond = pd.DataFrame(G_cond)
        G_New_tilde = G.groupby(A_group).sum() # G(A_tilde, L)
        G_New_tilde = np.array(G_New_tilde).astype(np.float64)
        G_tilde_cond = np.transpose(np.transpose(G_New_tilde)/np.sum(G_New_tilde,1)) # G(L|A_tilde)
        G = np.array(G).astype(np.float64)
        G_cond = np.array(G_cond).astype(np.float64)
    
        H_Boundary_U = np.sum(H, 1) # h(U)
        H_cond = np.transpose(np.transpose(H)/H_Boundary_U) # h(L|U)
        
        H = pd.DataFrame(H)
        H_cond = pd.DataFrame(H_cond)
        H_New_tilde = H.groupby(U_group).sum() # h(U_tilde, L)
        H_New_tilde = np.array(H_New_tilde).astype(np.float64)
        H_tilde_cond = np.transpose(np.transpose(H_New_tilde)/np.sum(H_New_tilde,1)) # G(L|A_tilde)
        H = np.array(H).astype(np.float64)
        H_cond = np.array(H_cond).astype(np.float64)

        if flag == 0:
            Start = time.clock()
            F_hat = F_New_to_F_hat(F_New, U_group, A_group, Row_number_F, Col_number_F)
            # .iloc is very very slow!!!
            
            print(2)
            Elapsed = time.clock() - Start
            print("Step 1:", Elapsed)
            TempA = F.sum(1)
            TempB = F_New.sum(1)
            TempC = F.sum(0)
            TempD = F_New.sum(0)
            # for the sake of convenience and speed
            
            U_cond_temp = np.random.uniform(0,1, Row_number_F)
            for i in range(Row_number_F):
                U_cond_temp[i] = TempA[i]/TempB[U_group[i]]
                
            U_cond = np.tile(U_cond_temp,(Col_number_F,1)).transpose()
    # np.tile: two numbers indicate the multiplication
            print(3)
            
            A_cond_temp = np.random.uniform(0,1, Col_number_F)
            for i in range(Col_number_F):
                A_cond_temp[i] = TempC[i]/TempD[A_group[i]]
                
            A_cond = np.tile(A_cond_temp,(Row_number_F,1))
            print(4)
    
            F_tilde = F_hat * U_cond * A_cond # f_tilde(U, A)
            flag = 1
            print(5)
            
        F_tilde = pd.DataFrame(F_tilde)
        C = F_tilde.groupby(U_group).sum()
        F_tilde_cond = np.transpose(np.transpose(C)/np.sum(C, 1)) # f_tilde(A | U_tilde)
        F_tilde_cond = np.array(F_tilde_cond).astype(np.float64)
        
        F_tilde_transpose = np.transpose(F_tilde)
        D = F_tilde_transpose.groupby(A_group).sum()
        F_tilde_transpose_cond = np.transpose(np.transpose(D)/np.sum(D, 1)) # f_tilde(A | U_tilde)
        F_tilde_transpose_cond = np.array(F_tilde_transpose_cond).astype(np.float64)
        F_transpose_cond = np.transpose(F_cond)
        
        # To compute the KL divergence between F and G, the transposed F is required for the clustering of A-group which originally occupies the columns
        # NOTE: np.transpose(F).values == np.transpose(np.array(F))
        
        Start = time.clock()
        
     #   F_cond = np.array(F_cond)
     #   F_tilde_cond = np.array(F_tilde_cond)
     #   F_tilde_transpose_cond = np.array(F_tilde_transpose_cond)
     #   F_transpose_cond = np.array(F_transpose_cond)
     #   G_cond = np.array(G_cond)
     #   G_tilde_cond = np.array(G_tilde_cond)
      #  H_cond = np.array(H_cond)
     #   H_tilde_cond = np.array(H_tilde_cond)
        
        F_cond = RemoveZero(F_cond)
        F_tilde_cond = RemoveZero(F_tilde_cond)
        F_tilde_transpose_cond = RemoveZero(F_tilde_transpose_cond)
        F_transpose_cond = RemoveZero(F_transpose_cond)
        G_cond = RemoveZero(G_cond)
        G_tilde_cond = RemoveZero(G_tilde_cond)
        H_cond = RemoveZero(H_cond)
        H_tilde_cond = RemoveZero(H_tilde_cond)
        
    #    F_cond = pd.DataFrame(F_cond)
    #    F_tilde_cond = pd.DataFrame(F_tilde_cond)
    #    F_tilde_transpose_cond = pd.DataFrame(F_tilde_transpose_cond)
    #    F_transpose_cond = pd.DataFrame(F_transpose_cond)
    #    G_cond = pd.DataFrame(G_cond)
    #    G_tilde_cond = pd.DataFrame(G_tilde_cond)
    #    H_cond = pd.DataFrame(H_cond)
    #    H_tilde_cond = pd.DataFrame(H_tilde_cond)
        
        '''
        F_cond[np.abs(F_cond <= 10**-11)] = 10**-11
    #    print(6)
        F_tilde_cond[np.abs(F_tilde_cond <= 10**-11)] = 10**-11
     #   print(7)
        F_tilde_transpose_cond[np.abs(F_tilde_transpose_cond <= 10**-11)] = 10**-11
        F_transpose_cond[np.abs(F_transpose_cond <= 10**-11)] = 10**-11
        G_cond[np.abs(G_cond <= 10**-11)] = 10**-11
        G_tilde_cond[np.abs(G_tilde_cond <= 10**-11)] = 10**-11
        H_cond[np.abs(H_cond <= 10**-11)] = 10**-11
        H_tilde_cond[np.abs(H_tilde_cond <= 10**-11)] = 10**-11
        '''
        End = time.clock()
        Elapsed = End - Start
        print("Step abs:", Elapsed)
    
        U_group_New = np.array(range(Row_number_F))
        A_group_New = np.array(range(Col_number_F))
        
        ## Step 1 for Optimization
        Start = time.clock()
        F_cond = np.array(F_cond).astype(np.float64)
        H_cond = np.array(H_cond).astype(np.float64)
        # How to do broadcast to prevent loop?
                
    #    F_tilde_cond = F_tilde_cond.values
        [Row_F_cond, Col_F_cond] = F_cond.shape
        F_cond = F_cond.reshape(1, Row_F_cond, Col_F_cond)
        [Row_F_tilde_cond, Col_F_tilde_cond] = F_tilde_cond.shape
        F_tilde_cond = F_tilde_cond.reshape(Row_F_tilde_cond, 1, Col_F_tilde_cond)
        TempSum1 = np.sum(F_cond * np.log10(F_cond / F_tilde_cond), 2)
        F_cond = F_cond.reshape(Row_F_cond, Col_F_cond)
        F_tilde_cond = F_tilde_cond.reshape(Row_F_tilde_cond, Col_F_tilde_cond)
    #    F_cond = pd.DataFrame(F_cond)
    #    F_tilde_cond = pd.DataFrame(F_tilde_cond)
        
   #     H_tilde_cond = H_tilde_cond.values
        [Row_H_cond, Col_H_cond] = H_cond.shape
        H_cond = H_cond.reshape(1, Row_H_cond, Col_H_cond)
        [Row_H_tilde_cond, Col_H_tilde_cond] = H_tilde_cond.shape
        H_tilde_cond = H_tilde_cond.reshape(Row_H_tilde_cond, 1, Col_H_tilde_cond)
        TempSum2 = np.sum(H_cond * np.log10(H_cond / H_tilde_cond), 2)
        H_cond = H_cond.reshape(Row_H_cond, Col_H_cond)
        H_tilde_cond = H_tilde_cond.reshape(Row_H_tilde_cond, Col_H_tilde_cond)
  #      H_cond = pd.DataFrame(H_cond)
   #     H_tilde_cond = pd.DataFrame(H_tilde_cond)
        
        '''
        Shape:
            F_Boundary_U: (9032,)
            TempSum1: (9, 9032)
            H_Boundary_U: (9032,)
            TempSum2: (9, 9032)
        '''
        F_Boundary_U = np.array(F_Boundary_U).astype(np.float64)
        H_Boundary_U = np.array(H_Boundary_U).astype(np.float64)
        Sum = F_Boundary_U * TempSum1 + H_Boundary_U * varphi * TempSum2
        # Important: Change pandas to numpy could lead to good results
        # ValueError: Length of passed values is 3, index implies 10
        Result = np.argmin(Sum, 0) 
        U_group_New = Result
        U_group = U_group_New.copy()
        End = time.clock()
        Elapsed = End - Start
        print("Step 2:", Elapsed)
    # Important for .copy() otherwise U_group will claim the same internal storage with U_group_New

    ## Step 2 for Optimization
        Start = time.clock()
    #    F_transpose_cond = np.array(F_transpose_cond)
    #    F_tilde_transpose_cond = np.array(F_tilde_transpose_cond)
        [Row_F_transpose_cond, Col_F_transpose_cond] = F_transpose_cond.shape
        F_transpose_cond = F_transpose_cond.reshape(1, Row_F_transpose_cond, Col_F_transpose_cond)
        [Row_F_tilde_transpose_cond, Col_F_tilde_transpose_cond] = F_tilde_transpose_cond.shape
        F_tilde_transpose_cond = F_tilde_transpose_cond.reshape(Row_F_tilde_transpose_cond, 1, Col_F_tilde_transpose_cond)
        TempSum1 = np.sum(F_transpose_cond * np.log10(F_transpose_cond / F_tilde_transpose_cond), 2)
        F_transpose_cond = F_transpose_cond.reshape(Row_F_transpose_cond, Col_F_transpose_cond)
        F_tilde_transpose_cond = F_tilde_transpose_cond.reshape(Row_F_tilde_transpose_cond, Col_F_tilde_transpose_cond)
    #    F_transpose_cond = pd.DataFrame(F_transpose_cond)
    #    F_tilde_transpose_cond = pd.DataFrame(F_tilde_transpose_cond)
        
        G_cond = np.asarray(G_cond).astype(np.float64)
        G_cond = np.array(G_cond).astype(np.float64)
        G_tilde_cond = np.array(G_tilde_cond).astype(np.float64)
        [Row_G_cond, Col_G_cond] = G_cond.shape
        G_cond = G_cond.reshape(1, Row_G_cond, Col_G_cond)
        [Row_G_tilde_cond, Col_G_tilde_cond] = G_tilde_cond.shape
        G_tilde_cond = G_tilde_cond.reshape(Row_G_tilde_cond, 1, Col_G_tilde_cond)
        TempSum2 = np.sum(G_cond * np.log10(G_cond / G_tilde_cond), 2)
        G_cond = G_cond.reshape(Row_G_cond, Col_G_cond)
        G_tilde_cond = G_tilde_cond.reshape(Row_G_tilde_cond, Col_G_tilde_cond)
   #    G_cond = pd.DataFrame(G_cond)
     #   G_tilde_cond = pd.DataFrame(G_tilde_cond)
        
        F_Boundary_A = np.array(F_Boundary_A).astype(np.float64)
        G_Boundary_A = np.array(G_Boundary_A).astype(np.float64)
        Sum = F_Boundary_A * TempSum1 + G_Boundary_A * Lambda * TempSum2
        Result = np.argmin(Sum, 0) 
        A_group_New = Result
        '''
        F_transpose_cond = np.array(F_transpose_cond)
        G_cond = np.array(G_cond)
        for i in range(Col_number_F):
   #         print(i)
            Temp2 = F_transpose_cond[i,]
        # Temp1 and F_tilde_cond occupy the same internal unit space.
        # Do not use Temp1[1] = ... if you do not want to change F_tilde_cond

            TempSum1 = np.sum(Temp2 * np.log10(Temp2 / F_tilde_transpose_cond), 1)
            
            Temp3 = G_cond[i,]
            TempSum2 = np.sum(Temp3 * np.log10(Temp3 / G_tilde_cond), 1)
            
            f = F_Boundary_A[i]
            
            # Wrong! Should be the boundary probablities with A, not U
            
            g = G_Boundary_A[i]
            Sum = f * TempSum1 + g * Lambda * TempSum2
            Result = np.argmin(Sum) 
            A_group_New[i] = Result
        '''
        A_group = A_group_New.copy()
        End = time.clock()
        Elapsed = End - Start
        print("Step 3:", Elapsed)
    #    F_transpose_cond = pd.DataFrame(F_transpose_cond)
     #   G_cond = pd.DataFrame(G_cond)
        # Compute the objective function
        Start = time.clock()
    #    F_New = F_New.values
        F_hat = F_New_to_F_hat(F_New, U_group, A_group, Row_number_F, Col_number_F)
    #    F_hat = pd.DataFrame(F_hat)
    #    F_New = pd.DataFrame(F_New)
        End = time.clock()
        Elapsed = End - Start
        print("Step 4:", Elapsed)

        TempA = F.sum(1)
        TempB = F_New.sum(1)
        TempC = F.sum(0)
        TempD = F_New.sum(0)
    # for the sake of convenience and speed
        TempA = np.array(TempA).astype(np.float64)
        TempB = np.array(TempB).astype(np.float64)
        U_cond_temp = np.random.uniform(0,1, Row_number_F)
        for i in range(Row_number_F):
            U_cond_temp[i] = TempA[i]/TempB[U_group[i]]
        
        U_cond = np.tile(U_cond_temp,(Col_number_F,1)).transpose()
        # np.tile: two numbers indicate the multiplication
        print(2)
        
        TempC = np.array(TempC).astype(np.float64)
        TempD = np.array(TempD).astype(np.float64)
        A_cond_temp = np.random.uniform(0,1, Col_number_F)
        for i in range(Col_number_F):
            A_cond_temp[i] = TempC[i]/TempD[A_group[i]]
    
        A_cond = np.tile(A_cond_temp,(Row_number_F,1))
        print(3)
    
        F_tilde = F_hat * U_cond * A_cond # f_tilde(U, A)
        
        TempA = H.sum(1)
        TempB = H_New_tilde.sum(1)
        TempC = G.sum(1)
        TempD = G_New_tilde.sum(1)
        
        F = pd.DataFrame(F)
        B = F.groupby(U_group).sum()
        F_New = B.transpose().groupby(A_group).sum().transpose() # f(U_tilde,A_tilde)
        F = np.array(F).astype(np.float64)
        F_New = np.array(F_New).astype(np.float64)


        H_New_tilde = np.array(H_New_tilde)
        H_hat = H_New_tilde_to_H_hat(H_New_tilde, U_group, Row_number_F, Col_number_H)
        # H_New_tilde = pd.DataFrame(H_New_tilde)
        U_cond_temp = np.random.uniform(0,1, Row_number_F)
        
        TempA = np.array(TempA).astype(np.float64)
        TempB = np.array(TempB).astype(np.float64)
        for i in range(Row_number_F):
            U_cond_temp[i] = TempA[i]/TempB[U_group[i]]
        U_cond_H = np.tile(U_cond_temp,(Col_number_H,1)).transpose()
    # np.tile: two numbers indicate the multiplication
        H_tilde = H_hat * U_cond_H

        G_New_tilde = np.array(G_New_tilde).astype(np.float64)
        G_hat = G_New_tilde_to_G_hat(G_New_tilde, A_group, Col_number_F, Col_number_G)
        # G_New_tilde = pd.DataFrame(G_New_tilde)
        A_cond_temp = np.random.uniform(0,1, Col_number_F)
        TempC = np.array(TempC).astype(np.float64)
        TempD = np.array(TempD).astype(np.float64)
        for i in range(Col_number_F):
            A_cond_temp[i] = TempC[i]/TempD[A_group[i]]
    
        A_cond_G = np.tile(A_cond_temp,(Col_number_G,1)).transpose()
        G_tilde = G_hat * A_cond_G
        
      #  H = np.array(H)
     #   H_tilde = np.array(H_tilde)
      #  G = np.array(G)
     #   G_tilde = np.array(G_tilde)
     #   F = np.array(F)
     #   F_tilde = np.array(F_tilde)
        if flag2 == 0:
            H = RemoveZero(H)
            H_tilde = RemoveZero(H_tilde)
            G = RemoveZero(G)
            G_tilde = RemoveZero(G_tilde)
            F = RemoveZero(F)
            F_tilde = RemoveZero(F_tilde)
            flag2 = 1
        '''
        H[np.abs(H <= 10**-11)] = 10**-11
        H_tilde[np.abs(H_tilde <= 10**-11)] = 10**-11
        G[np.abs(G <= 10**-11)] = 10**-11
        G_tilde[np.abs(G_tilde <= 10**-11)] = 10**-11
        F[np.abs(F <= 10**-11)] = 10**-11
        F_tilde[np.abs(F_tilde <= 10**-11)] = 10**-11
        '''
      #  H = pd.DataFrame(H)
      #  G = pd.DataFrame(G)
     #   F = pd.DataFrame(F)
     #   F_tilde = pd.DataFrame(F_tilde)
        
     #   print(I, "\n\n")
  #      print(type(I))
     #   print(I.shape, "\n\n")
        # print(np.log10(I), "\n")
      #  print(J)
        
        Sum_F = np.sum(F*np.log10(F/F_tilde)).sum()
        I = G/G_tilde
        I = np.asarray(I).astype(np.float64) # Important!
        J = np.log10(I)
        G = np.asarray(G).astype(np.float64) # Important!
     #   J = pd.DataFrame(J)
        Sum_G = np.sum(G*J).sum()
        Sum_H = np.sum(H*np.log10(H/H_tilde)).sum() 
        # Wrong! There exists negative elements.
        Q_New = Sum_F + Lambda*Sum_G + varphi*Sum_H
        print(np.abs(Q_New - Q))
        if(np.abs(Q_New - Q) <= 10**-2 or Q_New >= Q):
            print(Q)
            break
        Q = Q_New
        print(Q)
        Num = Num+1
    return U_group, A_group, User_Profile, User_Item, Item_Profile
    
def Visualization(r):
    TEMP1 = [i for i, x in enumerate(r[0]) if x == 0]
    TEMP2 = [i for i, x in enumerate(r[0]) if x == 1]
    TEMP3 = [i for i, x in enumerate(r[0]) if x == 2]
    TEMP4 = [i for i, x in enumerate(r[0]) if x == 3]
    TEMP5 = [i for i, x in enumerate(r[0]) if x == 4]
    TEMP6 = [i for i, x in enumerate(r[0]) if x == 5]
    TEMP7 = [i for i, x in enumerate(r[0]) if x == 6]
    TEMP8 = [i for i, x in enumerate(r[0]) if x == 7]
    TEMP9 = [i for i, x in enumerate(r[0]) if x == 8]
   # TEMP4 = [i for i, x in enumerate(r[1]) if x == 0]
  #  TEMP5 = [i for i, x in enumerate(r[1]) if x == 1]
  #  TEMP6 = [i for i, x in enumerate(r[1]) if x == 2]
 #   r[2].loc[r[2].index.isin(TEMP1)].to_csv('Cluster_User_1_subset_lambda400.csv')
#    r[2].loc[r[2].index.isin(TEMP2)].to_csv('Cluster_User_2_subset_lambda400.csv')
  #  r[2].loc[r[2].index.isin(TEMP3)].to_csv('Cluster_User_3_subset_lambda400.csv')
  #  r[3].loc[r[3].index.isin(TEMP1)].to_csv('Cluster_User_4_subset_lambda400.csv')
  #  r[3].loc[r[3].index.isin(TEMP2)].to_csv('Cluster_User_5_subset_lambda400.csv')
  #  r[3].loc[r[3].index.isin(TEMP3)].to_csv('Cluster_User_6_subset_lambda400.csv')
   # r[4].loc[r[4].index.isin(TEMP4)].to_csv('Cluster_Item_1_subset_lambda400.csv')
  # r[4].loc[r[4].index.isin(TEMP5)].to_csv('Cluster_Item_2_subset_lambda400.csv')
   # r[4].loc[r[4].index.isin(TEMP6)].to_csv('Cluster_Item_3_subset_lambda400.csv')
    # items in r[2] are one-order, while items in r[3] are two-order
    
    Group_1 = r[3].loc[r[3].index.isin(TEMP1)].sum(0).sort_values(ascending = False)
  #  Group_1.index = list(map(lambda x:x[2:],Group_1.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_1.index[1:20], Group_1[1:20])
    plt.savefig("Group_1_lambda=400")
    # When outputing a picture, do not execute the code by line
    Group_2 = r[3].loc[r[3].index.isin(TEMP2)].sum(0).sort_values(ascending = False)
 #   Group_2.index = list(map(lambda x:x[2:],Group_2.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_2.index[1:20], Group_2[1:20])
    plt.savefig("Group_2_lambda=400")
    Group_3 = r[3].loc[r[3].index.isin(TEMP3)].sum(0).sort_values(ascending = False)
#    Group_3.index = list(map(lambda x:x[2:],Group_3.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_3.index[1:20], Group_3[1:20])
    plt.savefig("Group_3_lambda=400")
    Group_4 = r[3].loc[r[3].index.isin(TEMP4)].sum(0).sort_values(ascending = False)
 #   Group_4.index = list(map(lambda x:x[2:],Group_4.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_4.index[1:20], Group_4[1:20])
    plt.savefig("Group_4_lambda=400")
    Group_5 = r[3].loc[r[3].index.isin(TEMP5)].sum(0).sort_values(ascending = False)
  #  Group_5.index = list(map(lambda x:x[2:],Group_5.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_5.index[1:20], Group_5[1:20])
    plt.savefig("Group_5_lambda=400")
    Group_6 = r[3].loc[r[3].index.isin(TEMP6)].sum(0).sort_values(ascending = False)
  #  Group_6.index = list(map(lambda x:x[2:],Group_6.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_6.index[1:20], Group_6[1:20])
    plt.savefig("Group_6_lambda=400")
    Group_7 = r[3].loc[r[3].index.isin(TEMP7)].sum(0).sort_values(ascending = False)
  #  Group_7.index = list(map(lambda x:x[2:],Group_7.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_7.index[1:20], Group_7[1:20])
    plt.savefig("Group_7_lambda=400")
    Group_8 = r[3].loc[r[3].index.isin(TEMP8)].sum(0).sort_values(ascending = False)
  #  Group_8.index = list(map(lambda x:x[2:],Group_8.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_8.index[1:20], Group_8[1:20])
    plt.savefig("Group_8_lambda=400")
    Group_9 = r[3].loc[r[3].index.isin(TEMP9)].sum(0).sort_values(ascending = False)
 #   Group_9.index = list(map(lambda x:x[2:],Group_9.index))
    plt.figure(figsize=(20,8),dpi=80)
    plt.bar(Group_9.index[1:20], Group_9[1:20])
    plt.savefig("Group_9_lambda=400")
    
    
if __name__ == '__main__':
    matplotlib.rcParams['font.family']='STSong'
    # For showing Chinese characters
    r = CCAM()
    Visualization(r)
    
