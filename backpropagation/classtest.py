# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)
print(X.shape, y.shape)
print(X,y)

axes = [-1.5, 2.5, -1, 1.5]


plt.plot(X[y==0][:,0], X[y==0][:,1], 'r.')
plt.plot(X[y==1][:,0], X[y==1][:,1], 'b.')
plt.grid()
plt.axis(axes)

svc = SVC(C=10)
svc.fit(X, y)


x0s = np.linspace(axes[0], axes[1] , 200)
x1s = np.linspace(axes[2], axes[3], 200)
x0, x1 = np.meshgrid(x0s, x1s)
X = np.c_[x0.ravel(), x1.ravel()]
y_pred = svc.predict(X)
y_pred = y_pred.reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)


plt.show()