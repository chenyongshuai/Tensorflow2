# -*- coding: utf-8 -*-
import numpy as np

result = np.arange(1,10,2)
print(result)
result = np.tile(np.arange(0, 1200), (398, 1))
print(result)
result = np.tile(np.arange(0, 398 * 480, 480), (1200, 1)).T
print(result)


indices = np.tile(np.arange(0, 1200), (398, 1)) + np.tile(np.arange(0, 398 * 480, 480), (1200, 1)).T
print(result)