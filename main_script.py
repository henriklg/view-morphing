from findfundmat import findfundmat
from prewarp import find_prewarp
import cv2
import numpy as np
from numpy.linalg import inv




image1 = cv2.imread('data/einstein1.jpg')
image2 = cv2.imread('data/einstein3.jpg')
F = findfundmat(image1, image2)
print('F: ', F)

H0, H1 = find_prewarp(F)


print('H0: ', H0)
print('H1: ', H1)

width, height, _ = image1.shape
dest = np.zeros(image1.shape)
dest = cv2.warpPerspective(image1, inv(H0), (width, height))
dest2 = cv2.warpPerspective(image2, inv(H0), (width, height))

while(True):
    cv2.imshow('wind', dest)
    cv2.imshow('wind2', dest2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break