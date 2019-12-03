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
def horizontalCutting(image):
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
    #cv2.imshow('canva',canva)
    #cv2.waitKey()
    

#垂直投影
def verticalCutting(image):
    image = np.asarray(image)
    canva = np.zeros(image.shape, np.uint8)
    (h, w) = image.shape
    w_ = [0] * w
    # 统计每列白色像素个数
    for i in range(w):
        for j in range(h):
            if image[j, i] == 255:
                w_[i] += 1
    # 绘制垂直投影
    for i in range(w):
        for j in range(w_[i]):
            canva[j, i] = 255
    #cv2.imshow('canva', canva)
    #cv2.waitKey()
    arrSplitIndex = []
    for i in range (0,len(canva[0])-1):
        if canva[0][i]>0 and canva[0][i-1]==0:
            arrSplitIndex.append(i-1)
        elif canva[0][i]>0 and canva[0][i+1]==0:
            arrSplitIndex.append(i+1)
    arrNumberSplit = []
    for i in range (0,len(arrSplitIndex)-1):
        arrNumberSplit.append(image[:,arrSplitIndex[i]:arrSplitIndex[i+1]])
    for i in range (0,len(arrNumberSplit),2):
        image = Image.fromarray(arrNumberSplit[i])
        image.show()



dir = "../img/number.png"
image = Image.open(dir)
image = grayHandle(image)
image = inversePixel(image)
#horizontalCutting(image)
verticalCutting(image)






