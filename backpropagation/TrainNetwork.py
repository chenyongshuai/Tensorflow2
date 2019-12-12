# -*- coding: utf-8 -*-
from sklearn.datasets import make_moons

from backpropagation.Layer import Layer
from backpropagation.NeuralNetwork import NeuralNetwork
from scikitlearn_make_moons import make_points
import numpy as np

nn = NeuralNetwork() # 实例化网络类
nn.add_layer(Layer(2, 25, 'sigmoid')) # 隐藏层 1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层 2, 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid')) # 隐藏层 3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid')) # 输出层, 25=>2

X_train,X_test,y_train,y_test = make_points()

output_lable,loss,acc = nn.train(X_train,X_test,y_train,y_test,0.001,1000)

nn.decisionboundary(X_test,output_lable,y_test.reshape(len(y_test),),[-1.5, 2.5, -1, 1.5])
