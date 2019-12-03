# -*- coding: utf-8 -*-
import sys

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout

def loadImage():
    arr_matrix = []
    for i in range (0,10):
        image = Image.open(f"./num{i}.png")
        image = image.convert("L")
        matrix = np.asarray(image)
        arr_matrix.append(matrix)

    #new_im = Image.fromarray(np.reshape(matrix, (28, 28)))
    #new_im.show()
    return arr_matrix

image_matrix = loadImage()

model = Sequential([Dense(256, activation='relu'),
                   Dense(128, activation='relu'),
                   Dense(28, activation='relu'),
                   Dense(10, activation='softmax')])
model.build(input_shape=(None, 28 * 28))
# 3、编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.load_weights('./mnist_model.h5')

weights = model.get_weights()

image_matrix = np.array(image_matrix)

image_matrix = image_matrix.reshape(10, 28*28).astype('float32') / 255.0

lables = [0,1,2,3,4,5,6,7,8,9]

image_labels = tf.keras.utils.to_categorical(lables)

image_lable = model.predict(image_matrix)

model.evaluate(image_matrix, image_labels)

#loss = model.train_on_batch(image_matrix,image_lable)

np.set_printoptions(suppress=True,threshold=sys.maxsize)

for i in range (0,10):
    print(image_lable[i][i])