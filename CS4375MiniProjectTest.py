# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 21:40:43 2025

@author: dache
"""


import scipy.io
import matplotlib.pyplot as plt
import numpy as np

mat = scipy.io.loadmat('98.mat')
print(mat)
print(mat["X098_DE_time"].shape[0] + mat["X098_FE_time"].shape[0])
#print(mat["X098_FE_time"])
#print(mat["X098RPM"])
'''mat2 = scipy.io.loadmat('99_NORMAL.mat')
print(mat2.keys())
print(mat2["X098_DE_time"])
print(mat2["X098_FE_time"])
#print(mat2["X098RPM"])
sampling_rate=12000
time=np.arange(mat["X097_DE_time"].shape[0])
time2=np.arange(mat2["X118_DE_time"].shape[0])
plt.plot(time, mat["X097_DE_time"], color="red")
plt.show()
plt.plot(time2, mat2["X118_DE_time"], color="blue")
plt.show()'''