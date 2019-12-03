# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import datasets
import os

# 1、创建数据集
(train_images,train_labels),(test_images,test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape(60000, 28*28).astype('float32') / 255.0
test_images = test_images.reshape(10000, 28*28).astype('float32') / 255.0
# 对标签进行分类编码
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 2、构建模型(顺序)
model = Sequential([Dense(256, activation='relu'),
                   Dense(128, activation='relu'),
                   Dense(28, activation='relu'),
                   Dense(10, activation='softmax')])
model.build(input_shape=(None, 28 * 28))
# 3、编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
# 4、拟合模型
history = model.fit(train_images, train_labels,
                    epochs=50, batch_size=512,
                    validation_data=(test_images, test_labels))
# 5、评估模型
model.evaluate(test_images, test_labels)

# 6、保存权重
os.remove('./mnist_model.h5')
model.save_weights('./mnist_model.h5')
#model.load_weights('./mnist_model.h5')
#predict = model.predict(test_images)
#loss = model.train_on_batch(test_images,test_labels)

# 结果可视化
acc = history.history.get('acc')
val_acc = history.history.get('val_acc')
loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

epochs = range(1, len(acc)+1)
plt.figure(figsize=(8,4),dpi=100)
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, '-b', label='Traing acc',linewidth=1 )
plt.plot(epochs, val_acc, '-r', label='Test acc',linewidth=1)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, '-b', label='Traing loss',linewidth=1)
plt.plot(epochs, val_loss, '-r', label='Test val_loss',linewidth=1)
plt.legend()
plt.show()