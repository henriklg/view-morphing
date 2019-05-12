from findfundmat import findfundmat
from prewarp import find_prewarp
from morph import automatic_point_correspondences, delaunay_triangulation
import cv2
import numpy as np
from numpy.linalg import inv

# Read image from file
image1 = cv2.imread('data/einstein1.jpg')
image2 = cv2.imread('data/einstein3.jpg')

# Find fundamental matrix
F = findfundmat(image1, image2)

# Get homographies from the fundamental matrix
H0, H1 = find_prewarp(F)

# Use homographies to warp images

new_size = int(np.sqrt(np.power(image1.shape[0], 2) + np.power(image1.shape[1], 2)))
print(new_size)
#width, height, _ = image1.shape
dest = cv2.warpPerspective(image1, inv(H0), (new_size, new_size))
dest2 = cv2.warpPerspective(image2, inv(H0), (new_size, new_size))

cv2.imshow('wind', dest)
cv2.imshow('wind2', dest2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Morph images
points1, points2 = automatic_point_correspondences(dest, dest2)
morph = delaunay_triangulation(dest, dest2, points1, points2)

cv2.imshow('morph', morph)

cv2.waitKey(0)
cv2.destroyAllWindows()