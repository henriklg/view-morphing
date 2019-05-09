from findfundmat import findfundmat
from prewarp import find_prewarp
import cv2
import numpy as np
from numpy.linalg import inv




image1 = cv2.imread('einstein1.jpg')
image2 = cv2.imread('einstein3.jpg')
F = findfundmat(image1, image2)
print('F: ', F)

F = np.array([[-0.0000,   0.0001,   -0.0182],
   [-0.0000,   0.0000,    0.0367],
    [0.0074,   -0.0527,   1.7703]])
H0,H1 = find_prewarp(F)

# H0 = np.array([[0.9881,    0.1540,    0.0004],
#    [-0.1540,    0.9881,   -0.0000],
#    [-0.0004,   -0.0001,    1.0000]])

print('H0: ', H0)
print('H1: ', H1)

width, height, _ = image1.shape
dest = np.zeros(image1.shape)
dest = cv2.warpPerspective(image1, inv(H0), (width, height))
dest2 = cv2.warpPerspective(image2, inv(H0), (width, height))
while(True):
    cv2.imshow('wind', dest)
    cv2.imshow('wind2', dest2)

    k = cv2.waitKey(20) & 0xFF
