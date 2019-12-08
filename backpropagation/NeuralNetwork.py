# -*- coding: utf-8 -*-
import numpy as np
class NeuralNetwork:
    #神经网络大类
    def __init__(self):
        self._layers = []#获取网络层列表
    def add_layer(self ,layer):
        self._layers.append(layer)#追加网络层
    def feed_forword(self, r):#前向传播循环调用网络层对象前向计算函数
        for layer in self._layers:
            X = layer.activition(r)#依次通过每个网络层激活函数
        return X
    def backpropagation(self, X , y, learning_rate):#反向传播算法
        output = self.feed_forword(X)#获取前向计算的输出值
        for i in reversed(range(len(self._layers))):#反向循环
            layer = self._layers[i]#获取反向循环当前网络层对象
            #如果是输出层
            if layer == self._layers[-1]:
                layer.error = y -output#2分类任务均方差导数
                #关键：计算最后一层的delta
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            #如果是隐藏层
            else:
                next_layer = self._layers[ i+1 ]
                layer.error = np.dot(next_layer.weights , next_layer.delta)










