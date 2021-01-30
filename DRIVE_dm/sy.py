from __future__ import division
import numpy as np 
from matplotlib import pyplot as plt 
from help_functions import *
import cv2

a = np.zeros((50,3,3,3),dtype=int)
b = np.ones((50,3,3,1),dtype=int)
c = [a,b]
x = c[0]
y = c[1]

# print(a.shape)
# print(x.shape)

print(y)
