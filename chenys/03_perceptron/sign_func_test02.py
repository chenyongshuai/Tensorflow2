# -*- coding: utf-8 -*-
#y = sign(x) x>=0 y=1    x<=0 y=-1
#x1(3,3) x2(4,3) y1,y2=1   x3(1,1) y3=-1
#f(x)=sign(∑a*y*xi*xj+b)
#误分类：yi*(∑aiyjxj*xi+b)<=0
import matplotlib.pyplot as plt
import numpy as np

def gram_cal(x):
    return x.dot(x.transpose())

def update(i,a,yi,bi,x_gram):
    α=1
    bi = bi + α*yi
    a[i]=a[i]+α
    print("更新参数：a: ",a.transpose(),", b: ",bi)
    a,bi = cal(y,bi,a,x_gram)
    return a,bi

def cal(y,b,a,x_gram):

    for i in range (0,len(x)):
        xi_gram=np.array(x_gram[i]).reshape(1, len(y))
        zi=np.dot(xi_gram, a * y ) + b
        re=0
        if zi[0][0] > 0:
            re=1
        elif zi[0][0]<0:
            re=-1
        if re ==y[i][0]:
            continue
        else:
            a,b = update(i,a,y[i][0],b,x_gram)#更新参数
            break
    return a, b



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

    w = np.dot(w.transpose(),x*y)
    x = np.linspace(0, 5, 50)  # 从(0,5)均匀取50个点
    #  w[0][0]*x + w[0][1]*y + b[0][0] = 0
    y = (-b[0][0]- w[0][0]*x)/w[0][1]

    plt.plot(x, y)
    plt.show()

x = np.array([[3, 3], [4, 3], [1, 1],[5,4]])
y = np.array([[1], [1], [-1], [1]])
a = np.array([[0], [0], [0],[0]])
b = np.array([[0]])
def main():

    x_gram = gram_cal(x)
    #print(x_gram[1].reshape(3,1).shape,a.shape, y.shape)
    #print(x_gram[1].reshape(3,1)*a * y)

    A,B= cal(y,b,a,x_gram)
    show(A,B,x,y)

main()

