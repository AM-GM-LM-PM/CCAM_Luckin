# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:02:24 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import numba
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from CCAM import CCAM


random_state = np.random.randint(1, 200, 1)[0]

@numba.jit
def JudgeZero(X):
    [m,n] = X.shape
    for i in range(m):
        for j in range(n):
            if(X[i,j] < 0):
                X[i,j] = -X[i,j]
    return X

def init():
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
    return User_Item, User_Profile, Item_Profile, F, H, G
    
def CombinationUser(F, H):
    Temp1 = [F, H]
    return np.concatenate(Temp1, axis = 1) # not pd.concat()!

def CombinationItem(F_transpose, G):
    Temp1 = [F_transpose, G]
    return np.concatenate(Temp1, axis = 1) # not pd.concat()!

def KMeansGenerateTrueLabel(F, H, G, n_clusters_user = 5, n_clusters_item = 3):
    UserValueInfo = CombinationUser(F, H)
   # return UserValueInfo
    F_transpose = np.transpose(F)
    ItemValueInfo = CombinationItem(F_transpose, G)
    # F_transpose: The value of Item-User matrix
    '''
    random_state = random_state
    Wrong! Use 
    global random_state
    to change its value
    '''
    UserTrueLabel = KMeans(n_clusters = n_clusters_user, random_state = random_state).fit_predict(UserValueInfo)
    ItemTrueLabel = KMeans(n_clusters = n_clusters_item, random_state = random_state).fit_predict(ItemValueInfo)
    return UserTrueLabel, ItemTrueLabel
    
def KMeansGeneratePredictedLabel(F, H, G, n_clusters_user = 5, n_clusters_item = 3):
    F_transpose = np.transpose(F)
    UserPredictLabel = KMeans(n_clusters = n_clusters_user, random_state = random_state).fit_predict(F)
    ItemPredictLabel = KMeans(n_clusters = n_clusters_item, random_state = random_state).fit_predict(F_transpose)
    return UserPredictLabel, ItemPredictLabel

def CorrelationHeat(data):
    names = ["(0, 'BOSS午餐')", "(0, '健康轻食')", "(0, '小鹿茶')", "(0, '幸运小食')",
       "(0, '现磨咖啡')", "(0, '瑞纳冰')", "(0, '经典饮品')", "(0, '餐厨水具')",
       "(0, '鲜榨果蔬汁')", "(0, 'aaaaaaaaa')", "(0, 'afternoon')",
       "(0, 'evening')", "(0, 'morning')", "(0, 'night')", "(0, 'noon')",
       "sex", "property", "receive_sms"]
    correlations = data.corr()
    correction = abs(correlations)
    fig, ax = plt.subplots(figsize=(40, 40))
    ax = sns.heatmap(correction,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
    plt.savefig("Gfsdfsdf")
    
if __name__ == "__main__":
    os.chdir(os.getcwd())
    User_Item, User_Profile, Item_Profile, F, H, G = init()
    CorrelationHeat(User_Profile)
    UserTrueLabel, ItemTrueLabel = KMeansGenerateTrueLabel(F, H, G)
    UserPredictLabelKMeans, ItemPredictLabelKMeans = KMeansGeneratePredictedLabel(F, H, G)
    # UserTrueLabel = np.sort(UserTrueLabel)
    # UserPredictLabel = np.sort(UserPredictLabel)
    # print(UserPredictLabelKMeans == UserTrueLabel)
    UserPredictLabelCCAM, ItemPredictLabelCCAM, *args, F, G, H = CCAM(n_clusters_user = 5, n_clusters_item = 3)
    print(confusion_matrix(UserTrueLabel, UserPredictLabelKMeans))
    print(confusion_matrix(UserTrueLabel, UserPredictLabelCCAM))
    print(confusion_matrix(ItemTrueLabel, ItemPredictLabelCCAM))
    print(confusion_matrix(ItemTrueLabel, ItemPredictLabelKMeans))
    
