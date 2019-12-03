# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import sys
import cv2

np.set_printoptions(suppress=True,threshold=sys.maxsize)

#灰度处理
def grayHandle(image):
    image = image.convert("L")
    return image

#黑白反转
def inversePixel(image):
    arrImage = np.asarray(image)
    arrImage = 255-arrImage
    image = Image.fromarray(arrImage)
    return image

#水平投影
def horizontalProjection(image):
    image = np.asarray(image)
    canva = np.zeros(image.shape,np.uint8)
    (h,w) = image.shape
    h_ = [0]*h
    #统计每行白色像素个数
    for i in range (h):
        for j in range (w):
            if image[i,j] == 255:
                h_[i] += 1
    #绘制水平投影
    for i in range (h):
        for j in range (h_[i]):
            canva[i,j] = 255
    cv2.imshow('canva',canva)
    cv2.waitKey()

#垂直投影
def verticalProjection(image):
    image = np.asarray(image)
    canva = np.zeros(image.shape, np.uint8)
    (h, w) = image.shape
    w_ = [0] * w
    # 统计每行白色像素个数
    for i in range(h):
        for j in range(w):
            if image[j, i] == 255:
                w_[i] += 1
    # 绘制水平投影
    for i in range(w):
        for j in range(w_[i]):
            canva[i, j] = 255
    cv2.imshow('canva', canva)
    cv2.waitKey()


dir = "../img/number.png"
image = Image.open(dir)
image = grayHandle(image)
image = inversePixel(image)
horizontalProjection(image)
verticalProjection(image)






