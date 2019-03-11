# -*- coding: utf-8 -*-

"""
Created on Tue Jan 29 13:31:53 2019

@author: rv17643
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings



x = 'book1.csv' 
numerical = []
categorical=[]
a = open(x,'r')
b = a.read()
a = a.close()
b = b.splitlines()

for i in range(len(b)):
    b[i] = b[i].strip().split(',')
    
    if b[i][1] == "Categorical":
        categorical = categorical+[(b[i][0]).strip()]
    elif b[i][1] == "Numerical":
        numerical = numerical+[(b[i][0]).strip()]
    else:
        print(b[i])


y = 'train.csv'
df_train = pd.read_csv(y)
for i in numerical:
    
    data = pd.concat([df_train['SalePrice'], df_train[i]], axis=1)
    data.plot.scatter(x=i, y='SalePrice', ylim=(0,800000));

for i in categorical:
    data = pd.concat([df_train['SalePrice'], df_train[i]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=i, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);
    plt.xticks(rotation=90);
