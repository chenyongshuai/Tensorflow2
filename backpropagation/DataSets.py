# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import *

class DataSets:

#1.将数据集拆分成训练集和测试集
    def split_dataset(self, X, y, train_p=0.6, test_p=0.4, valid_p=None ,shuffle = True):
        print(X,y,shuffle,train_p,test_p,valid_p)
        train_set = []
        test_set = []
        valid_set = []
        train_lable = []
        test_lable = []
        valid_lable = []
        l = len(X)
        i1 = int(l * train_p)
        i2 = int(l * (train_p + test_p))
        print(i1,i2)
        if valid_p is None or valid_p==0:
            train_set = X[0:i1, :]
            test_set = X[i1:i2, :]
            train_lable = y[0:i1]
            test_lable = y[i1:i2]
        else:
            train_set = X[0:i1, :]
            test_set = X[i1:i2, :]
            valid_set = X[i2:l,:]
            train_lable = y[0:i1]
            test_lable = y[i1:i2]
            valid_lable = y [i2:l]
        return train_set,train_lable,test_set,test_lable,valid_set,valid_lable




#2.将数据集拆分成训练集、测试集和验证集





#3

X, y = make_moons(n_samples=10, noise=0.2, random_state=100)

d = DataSets()
train_set,train_lable,test_set,test_lable,valid_set,valid_lable  = d.split_dataset(X, y ,0.7,0.2,0.1)

print(train_set,'\n',train_lable,'\n',test_set,'\n',test_lable,'\n',valid_set,'\n',valid_lable)
