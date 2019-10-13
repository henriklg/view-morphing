from pointCorrespondences import getPointCorrespondences, automatic_point_correspondences
from morph import delaunay_triangulation
import cv2

# Variables
im1 = cv2.imread('data/trump_bear.jpg')
im2 = cv2.imread('data/bear.jpg')
filename = 'trump_bear'
step_size = 0.1

# check if images are the same size
assert im1.shape == im2.shape, "Images does not have the same shapes."

# Get points
points_1, points_2 = getPointCorrespondences(im1, im2)  # [[x y] [x y]]

# Morph with delaunay triangulation
morph = delaunay_triangulation(im1, im2, points_1, points_2, 0)

i = 0
count = 0
while True:
    cv2.imshow("Output", morph)
    k = cv2.waitKey(20) & 0xFF

    # N: next worph
    if k == ord('n'):
        i += step_size
        if i > 1:
            i = 1
        morph = delaunay_triangulation(im1, im2, points_1, points_2, i)
        print("i: {:.2f}".format(i))

    # P: previous morph
    if k == ord('p'):
        i -= step_size
        if i < 0:
            i = 0
        morph = delaunay_triangulation(im1, im2, points_1, points_2, i)
        print("i: {:.2f}".format(i))

    # S: save image
    if k == ord('s'):
        cv2.imwrite('morphs/{}_morph{}.jpg'.format(filename, count), morph)
        count += 1
        print('saving: {}_morph{}.jpg'.format(filename, i))

    # Q: quit program
    if k == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
