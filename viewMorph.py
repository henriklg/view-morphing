import numpy as np
import cv2
from skimage.transform import warp
from findfundmat import findfundmat
from prewarp import find_prewarp

image1 = 'einstein1.jpg'
image2 = 'einstein3.jpg'
im1 = cv2.imread(image1,0)

F = findfundmat(image1, image2)
print('F =', F)

[H1, H2] = find_prewarp(F)
print('H1 =', H1)
print('H2 =', H2)


warped = warp(im1, H1)
cv2.imshow('image', warped)
cv2.waitKey(0)