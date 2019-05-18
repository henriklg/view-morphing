from find_fundamental import fundamental_matrix
from prewarp import find_prewarp
from morph import delaunay_triangulation, transform_points
from pointCorrespondences import automatic_point_correspondences, getPointCorrespondences
from numpy.linalg import inv
from Postwarp import getRectangle, getLines, homography, getPoints, homography_points
import cv2
import numpy as np

"""
    View morph example using images of Einstein
"""

# Read image from file
image1 = cv2.imread('data/einstein1.jpg')
image2 = cv2.imread('data/einstein3.jpg')
#
# newX, newY = image1.shape[1]*0.5, image1.shape[0]*0.5
# image1 = cv2.resize(image1, (int(newX),int(newY)))
# image2 = cv2.resize(image2, (int(newX),int(newY)))



# Find fundamental matrix
F = fundamental_matrix(image1, image2)

# Get homographies from the fundamental matrix
H0, H1 = find_prewarp(F)

points1, points2 = automatic_point_correspondences(image1, image2, returntype='list')


# Use homographies to warp images
new_size = int(np.sqrt(np.power(image1.shape[0], 2) + np.power(image1.shape[1], 2)))
prewarp_1 = cv2.warpPerspective(image1, H0, (new_size, new_size))
prewarp_2 = cv2.warpPerspective(image2, H1, (new_size, new_size))
print(prewarp_2)


# Save result
cv2.imwrite('prewarp1.png', prewarp_1)
cv2.imwrite('prewarp2.png', prewarp_2)

# Show result on screen. Click on any button to continue
cv2.imshow('Prewarped image 1', prewarp_1)
cv2.imshow('Prewarped image 2', prewarp_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morph the prewarped images
points1, points2 = automatic_point_correspondences(prewarp_1, prewarp_2)
print(points2)

#points1, points2 = automatic_point_correspondences(prewarp_1, prewarp_2)
print(len(points1), len(points2))
morphshape = np.shape(image1)
morph = delaunay_triangulation(prewarp_1, prewarp_2, points1, points2, morphshape, removepoints=True)

# save result
cv2.imwrite('morph.png', morph)

# Show result on screen
cv2.imshow('morph', morph)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Postwarp. Select point correspondences manually using mask, find homography and morph
m_points = getPoints(morph)
im = cv2.imread('data/mask_einstein.jpg')
p_points = getPoints(np.array(im).astype(np.uint8))
H_s = homography_points(m_points, p_points)
h, w, _ = image1.shape
final_morph = cv2.warpPerspective(morph, H_s, (h, w))

cv2.imwrite('final_morph.png', final_morph)
cv2.imshow('window', final_morph)

cv2.waitKey(0)
cv2.destroyAllWindows()


