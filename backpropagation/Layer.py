# -*- coding: utf-8 -*-
import numpy as np
#类 Layer 实现一个网络层，需要传入网络层的数据节点数，输出节点数，激活函数类型等参数，权值 weights 和偏置张量 bias 在初始化时根据输入、输出节点数自动生成并初始化：
class Layer:
    #全连接网络层的初始化方法
    def __init__(self, n_input, n_neurons, activition=None, weights=None, bias=None):
        """
        :param n_input: 输入节点个数
        :param n_neurons:  输出节点个数
        :param activition:  激活函数类型
        :param weights:  权值张量，默认类内部生成
        :param bias:  偏差因子，默认类内部生成
        np.random.randn:返回具有标准正态分布的样本
        np.sqrt:计算数组各元素的平方根
        """
        self.weights = weights if weights is not None else np.random.randn(n_input,n_neurons) * np.sqrt(1/n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activition = activition#激活函数类型
        self.last_activition = None#激活函数的输出值
        self.error = None#用于计算当前层delta变量的中间变量
        self.delta = None#记录当前层delta变量，用于梯度计算
    #实现网络层的前向传播
    def activate(self,x):
        r = np.dot(x , self.weights) + self.bias# r = W@X+b
        self.last_activition = self._apply_activation(r) #获取全连接层的输出
        return self.last_activition
    #计算激活函数的输出
    def _apply_activation(self,r):
        if self.activition is None:
            return r #无激活函数返回r
        elif self.activition == 'relu':
            return np.maximum(r,0)
        elif self.activition == 'tanh':
            return np.tanh(r)
        elif self.activition == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r
    #针对不同的激活函数，计算导数
    def apply_activation_derivative(self,r):
        if self.activition is None:
            return np.ones_like(r)
        elif self.activition == 'relu':
            grad = np.array(r, copy=True)
            grad[ r>0 ] = 1
            grad[ r<=0 ] = 0
            return grad
        elif self.activition == 'tanh':
            return 1 - r**2
        elif self.activition == 'sigmoid':
            return r * (1 - r)
        return r

        












