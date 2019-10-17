# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:24:22 2019

User Behavior Analysis and Commodity Recommendation for Point-Earning Apps
Yu-Ching Chen, Chia-Ching Yang, Yan-Jian Liau, Chia-Hui Chang
National Central University
Taoyuan, Taiwan
TAAI2016

Initial_Data_Process_Code
With Time Series Feature

@author: Administrator
"""

import numpy as np
import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# from statsmodels.graphics.factorplots import interaction_plot
# import matplotlib.pyplot as plt
# from scipy import stats
# import datetime as dt

Order_Data_Part1_Name = "122532-2019-09-10+152846.csv"
# The Order data from 2019-06-01 to 2019-07-20
Order_Data_Part2_Name = "122589-2019-09-10+163353.csv"
# The Order data from 2019-07-21 to 2019-08-31
Order_Data_Part1 = pd.read_csv(Order_Data_Part1_Name, encoding = 'utf-8', sep = ',', error_bad_lines=False)
Order_Data_Part2 = pd.read_csv(Order_Data_Part2_Name, encoding = 'utf-8')
# The data

objs = [Order_Data_Part1, Order_Data_Part2]
Order_Data = pd.concat(objs, axis=0)
# Combine two parts of data together

A = Order_Data.groupby(['phone_no'])['phone_no'].count().sort_values(ascending = False)

# B = A[(A >= 100) and (A <= 200)]
# This is wrong!
A1 = A >= 5
A2 = A <= 10
B = A[A1 & A2]
C = B.index
Filtered_Order_Data = Order_Data.loc[Order_Data['phone_no'].isin(C)] # .isin: select rows with specific row names
# Filter people with more than 20 buying items

Subset_Filtered_Order_Data_Commodity_Code = Filtered_Order_Data[['phone_no', 'two_category_name']]
Subset_Filtered_Order_Data_Commodity_Code = Subset_Filtered_Order_Data_Commodity_Code.reset_index(drop=True)
# Important

DT = pd.to_datetime(Filtered_Order_Data['dt'])
DT = DT.reset_index(drop = True)
DT = pd.DataFrame(DT) # Important
DT['day'] = pd.DataFrame([pd.to_datetime(DT.loc[i,'dt']).day for i in range(len(DT))])
DT['month'] = pd.DataFrame([pd.to_datetime(DT.loc[i,'dt']).month for i in range(len(DT))])
DT['day'] = pd.DataFrame([1 if DT.loc[i,'day'] >= 15 else 0] for i in range(len(DT)))

def MonthTrans(x, y):
    if x == 7:
        return y+2
    elif x == 8:
        return y+4
    else:
        return y

DT['day'] = pd.DataFrame([MonthTrans(DT.loc[i,'month'],DT.loc[i,'day']) for i in range(len(DT))])
# Create a function, combine it with list generator.

D = pd.DataFrame((DT['day']+1)/6, columns = None )
Result_temp = pd.concat([Subset_Filtered_Order_Data_Commodity_Code, D], axis = 1)
User_Item_Table = pd.pivot_table(Result_temp, index = 'phone_no', columns = 'two_category_name', aggfunc = np.sum, values = 'day', fill_value = 0)
# The final table for user-item
# Why need D exist?

User_Item_Table.to_csv('User_Item_subset2.csv', sep=',', header=True, index=True)

# from svmutil import *
# from svm import *
# y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 4 -b 1')
# model = svm_train(prob, param)
# yt = [1]
# xt = [{1: 1, 2: 1}]
# p_label, p_acc, p_val = svm_predict(yt, xt, model)
# print(p_label)

Finish_Time_Data_Name = "123859-2019-09-11+164113.csv"
# Table recorded finish_time of each phone_no

Finish_Time_Data = pd.read_csv(Finish_Time_Data_Name, encoding = 'utf-8')

# Result_temp2.isna().sum()
# In total 84817 missing data

Filtered_Finish_Time_Data = Finish_Time_Data.loc[Finish_Time_Data['phone_no'].isin(C)]
Subset_Finish_Time_Data = Filtered_Finish_Time_Data.sort_values(by = ['phone_no', 'finish_time'])
Subset_Finish_Time_Data = Subset_Finish_Time_Data.fillna(method = 'ffill')
Subset_Finish_Time_Data = Subset_Finish_Time_Data.sort_index()
# The nan data has been refilled by the following one time value.

Subset_Finish_Time_Data = Subset_Finish_Time_Data.reset_index(drop = True)
Subset_Finish_Time_Data['Finish_time'] = pd.to_datetime(Subset_Finish_Time_Data['finish_time'])
Subset_Finish_Time_Data = Subset_Finish_Time_Data.drop('finish_time', axis = 1)
Temp = np.tile("aaaaaaaaa", (len(Subset_Finish_Time_Data), 1))
# Create an array where each member has 9-digits capacity.
for i in range(len(Subset_Finish_Time_Data)):
    temp = Subset_Finish_Time_Data['Finish_time'][i].hour
    if temp >= 7 and temp <= 11:
        Temp[i] = "morning"
    elif temp >= 12 and temp <= 13:
        Temp[i] = "noon"
    elif temp >= 14 and temp <= 16:
        Temp[i] = "afternoon"
    elif temp >= 17 and temp <= 18:
        Temp[i] = "evening"
    elif temp >= 19 and temp <= 22:
        Temp[i] = "night"

Subset_Finish_Time_Data['DATE2'] = Temp
# Create new features related to date

Subset_Filtered_Order_Data_User_Two_Category = Filtered_Order_Data[['phone_no', 'two_category_name']]

def sub_max(arr, n):
    ARR = np.zeros(len(arr))
  #  print(ARR)
    for i in range(n):
        arr_ = arr
        Index = np.argmax(arr_)
   #     print(Index)
        ARR[Index] = 1
        arr_[np.argmax(arr_)] = np.min(arr)
        arr = arr_
    return ARR

E = np.tile(1, (len(Subset_Filtered_Order_Data_User_Two_Category), 1))
E = pd.DataFrame(E)
Subset_Filtered_Order_Data_User_Two_Category = Subset_Filtered_Order_Data_User_Two_Category.reset_index(drop=True)
Subset_Filtered_Order_Data_User_Two_Category = pd.concat([Subset_Filtered_Order_Data_User_Two_Category, E], axis = 1)
User_Two_Category_Item = pd.pivot_table(Subset_Filtered_Order_Data_User_Two_Category, index = 'phone_no', columns = 'two_category_name', aggfunc = np.sum, fill_value = 0)

for i in range(len(User_Two_Category_Item)):
    Value = User_Two_Category_Item.iloc[i].values
    ARR = sub_max(Value, 4)
    User_Two_Category_Item.iloc[i] = ARR
    
#User_Two_Category_Item.to_csv('User_Two_Category_Item.csv', sep=',', header=True, index=True)
# Select the bought two_category items of each member

Subset_Finish_Time_Data_Time_Consumption = Subset_Finish_Time_Data[['phone_no', 'DATE2']]

E = np.tile(1, (len(Subset_Finish_Time_Data_Time_Consumption), 1))
E = pd.DataFrame(E)
Subset_Finish_Time_Data_Time_Consumption =  Subset_Finish_Time_Data_Time_Consumption.reset_index(drop=True)
Subset_Finish_Time_Data_Time_Consumption = pd.concat([Subset_Finish_Time_Data_Time_Consumption, E], axis = 1)
User_Time_Consumption_Item = pd.pivot_table(Subset_Finish_Time_Data_Time_Consumption, index = 'phone_no', columns = 'DATE2', aggfunc = np.sum, fill_value = 0)
# User_Time_Consumption_Item = User_Time_Consumption_Item.drop((0,'aaaaaaaaa'), axis = 1)
# Drop the additional useless feature aaaaaaaaa

for i in range(len(User_Time_Consumption_Item)):
    Value = User_Time_Consumption_Item.iloc[i].values
    ARR = sub_max(Value, 2)
    User_Time_Consumption_Item.iloc[i] = ARR
# Indicate the largest two elements as 1 and otherwise 0.

User_Time_Consumption_Item.to_csv('User_Time_Consumption_Item_subset.csv', sep=',', header=True, index=True)

Result_temp = pd.concat([User_Two_Category_Item, User_Time_Consumption_Item], axis = 1)

User_Info_Data_Name = "124822-2019-09-12+152056.csv"
# Basic user info
User_Info_Data = pd.read_csv(User_Info_Data_Name, encoding = 'utf-8')
User_Info_Data = User_Info_Data.drop('id', axis = 1)

Filtered_User_Info_Data = User_Info_Data[User_Info_Data['phone_no'].isin(C)] # .isin: select rows with specific row names

# G = Filtered_Order_Data.index
# Filtered_User_Info_Data = Filtered_User_Info_Data.loc[Filtered_User_Info_Data['phone_no'].isin(G)] # .isin: select rows with specific row names
Filtered_User_Info_Data = Filtered_User_Info_Data.drop_duplicates(subset=None, keep='first', inplace=False)
Filtered_User_Info_Data = Filtered_User_Info_Data.reset_index(drop=True)
Filtered_User_Info_Data.sort_values(by = 'phone_no')
Filtered_User_Info_Data['sex'] -= 1
Filtered_User_Info_Data['property'] -= 1


User_Profile = pd.merge(Result_temp, Filtered_User_Info_Data, on='phone_no')
User_Profile = User_Profile.fillna(method = 'ffill')
# User_Profile = User_Profile.drop('id', axis = 1)


User_Profile.to_csv('User_Profile_subset2.csv', sep=',', header=True, index=True)

Item_Profile_Data = Filtered_Order_Data[['commodity_code', 'two_category_name']]
Item_Profile_Data = Item_Profile_Data.reset_index(drop=True)
# D = np.tile(1, (len(Item_Profile_Data), 1))
# D = pd.DataFrame(D)
Item_Profile_Data =  Item_Profile_Data.reset_index(drop=True)
Item_Profile_Data = pd.concat([Item_Profile_Data, D], axis = 1)
Item_Profile = pd.pivot_table(Item_Profile_Data, index = 'two_category_name', columns = 'commodity_code', aggfunc = np.sum, fill_value = 0)

Item_Profile.to_csv('Item_Profile_subset2.csv', sep=',', header=True, index=True)

# Item_Profile.values: shows the data matrix without row and column

