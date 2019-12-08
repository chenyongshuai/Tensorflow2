# -*- coding: utf-8 -*-
#y = sign(x) x>=0 y=1    x<=0 y=-1
#x1(3,3) x2(4,3) y1,y2=1   x3(1,1) y3=-1
#z = wX+b
#y^= sign(z)
#w: = w - α dw
#   = w - α * y * Xi
#b: = b - α * y
import matplotlib.pyplot as plt
import numpy as np



def update(wi,xi,yi,bi):
    α=1
    wi = wi + α*xi.dot(yi)
    bi = bi + α*yi
    print("更新参数：w: ",wi,", b: ",bi)
    wi,bi = cal(x,y,wi,bi)
    return wi,bi

def cal(x,y,w,b):
    for i in range (0,len(x)):
        zi=w.dot(x[i].reshape(2,1))+b
        re=0
        if zi[0][0] >=0:
            re=1
        else:
            re=-1
        if re ==y[i][0]:
            continue
        else:
            w,b = update(w,x[i].reshape(2,1),y[i],b)#更新参数
            break
    return w, b



def show(w,b,x,y):
    x0List = []
    y0List = []
    x1List = []
    y1List = []
    for i in range(0, x.__len__()):
        if y[i, :][0] >= 1:
            x0List.append(float(x[i, :][0]))
            y0List.append(float(x[i, :][1]))
        else:
            x1List.append(float(x[i, :][0]))
            y1List.append(float(x[i, :][1]))
    xList = []
    yList = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xList, yList)
    ax.scatter(x0List, y0List, s=10, c='blue')
    ax.scatter(x1List, y1List, s=10, c='black')


    print(w,w.shape,b,b.shape)
    x = np.linspace(0, 5, 50)  # 从(0,5)均匀取50个点
    #  w[0][0]*x + w[0][1]*y + b[0][0] = 0
    y = (-b[0][0]- w[0][0]*x)/w[0][1]

    plt.plot(x, y)
    plt.show()

x = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([[1], [1], [-1]])
w = np.array([[0, 0]])
b = np.array([[0]])
def main():
    W,B= cal(x,y,w,b)
    show(W,B,x,y)

main()

