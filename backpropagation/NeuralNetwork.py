# -*- coding: utf-8 -*-
import numpy as np
class NeuralNetwork:
    #神经网络大类
    def __init__(self):
        self._layers = []#获取网络层列表
    def add_layer(self ,layer):
        self._layers.append(layer)#追加网络层
    def feed_forword(self, r):#前向传播循环调用网络层对象前向计算函数
        for i in range(len(self._layers)):
            if i > 0 :
                r = self._layers[i-1].last_activition
            X = self._layers[i].activate(r)
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
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i 为上一网络层的输出
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            # 梯度下降算法，delta 是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate
    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        # 网络训练函数
        # one-hot 编码
        y_onehot = np.zeros((y_train.shape[0], 2))
        for i, j in zip(range(len(y_train)),y_train):
            y_onehot[i][j]=1
        mses = []
        for i in range(max_epochs):  # 训练 1000 个 epoch
            self.backpropagation(X_train, y_onehot, learning_rate)
            #for j in range(len(X_train)):  # 一次训练一个样本
            #    print(X_train[j], y_onehot[j])
            #    print(X_train[j].shape, y_onehot[j].shape)
            #    self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
            # 打印出 MSE Loss
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                # 统计并打印准确率
                #print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test),y_test.flatten()) * 100))
        return mses














