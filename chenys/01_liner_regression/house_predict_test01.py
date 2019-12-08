# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#读取数据
data = pd.read_csv('houseSet.txt',header=None,names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT','MEDV'],sep='\s+')
#查看数据是否有空值
data.isnull().any().sum()
#查看数据分布图
#pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(10,10), diagonal='kde')
#查看特征相关性大小
corr = data.corr()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',200)
pd.set_option('expand_frame_repr', False)
corr.sort_values(by=['MEDV']).sort_index(axis=0,ascending=True)
#选择特征以及特征分布图
features = data[['RM', 'PTRATIO', 'LSTAT']]
#pd.plotting.scatter_matrix(features, alpha=0.7, figsize=(6,6), diagonal='hist')
#plt.show()

features_desc = features.describe()
#RM=(np.mat(features['RM']))
#PTRATIO=(np.mat(features['PTRATIO']))/100
#LSTAT=(np.mat(features['LSTAT']))/100
#特征值样本值归一化  X=(X-Xmin) / (Xmax-Xmin)
RM=(np.mat(features['RM'])-features_desc['RM']['min'])/(features_desc['RM']['max']-features_desc['RM']['min'])
PTRATIO=(np.mat(features['PTRATIO'])-features_desc['PTRATIO']['min'])/(features_desc['PTRATIO']['max']-features_desc['PTRATIO']['min'])
LSTAT=(np.mat(features['LSTAT'])-features_desc['LSTAT']['min'])/(features_desc['LSTAT']['max']-features_desc['LSTAT']['min'])

#RM1=(features['RM']-features_desc['RM']['min'])/(features_desc['RM']['max']-features_desc['RM']['min'])
#PTRATIO1=(features['PTRATIO']-features_desc['PTRATIO']['min'])/(features_desc['PTRATIO']['max']-features_desc['PTRATIO']['min'])
#LSTAT1=(features['LSTAT']-features_desc['LSTAT']['min'])/(features_desc['LSTAT']['max']-features_desc['LSTAT']['min'])

#6.575     15.3   4.98  0.577505    0.287234  0.089680
#scaler = MinMaxScaler()
#for feature in features.columns:
#    features['标准化'+feature] = scaler.fit_transform(features.loc[[feature]])
#print(features)


#dataMatrix_after = pd.DataFrame({'a':RM1,'b':PTRATIO1,'c':LSTAT1})
#pd.plotting.scatter_matrix(dataMatrix_after, alpha=0.7, figsize=(6,6), diagonal='hist')
#plt.show()


B=np.ones([1,506])
dataMatrix = np.vstack((RM,PTRATIO,LSTAT,B)).transpose()
#MEDV=(np.mat(features['MEDV']).transpose()-features_desc['MEDV']['min'])/(features_desc['MEDV']['max']-features_desc['MEDV']['min'])
MEDV=np.mat(data['MEDV']).transpose()
m, n = np.shape(dataMatrix)
matMatrix = np.mat(dataMatrix)
w = np.ones((n, 1))
alpha = 0.001
num = 500
for i in range(num):
    # dz = a - y
    error = (matMatrix * w -MEDV)
    # w = w - α * (x * dz)
    w = w - alpha * matMatrix.transpose() * error

print(w)
W=np.transpose(w[0:3,:])
b=np.transpose(w[3:4,:])

features_test=data[['RM', 'PTRATIO', 'LSTAT']]
scaler = MinMaxScaler()
for feature in features_test.columns:
    features_test[feature] = scaler.fit_transform(features_test[[feature]])


Y=W.dot(features_test.transpose())+b
print(Y)