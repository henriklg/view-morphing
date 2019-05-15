from pointCorrespondences import getPointCorrespondences
from morph import delaunay_triangulation, automatic_point_correspondences
import cv2

# Load two images of same size
im1 = cv2.imread('data/trump_bear.jpg')
im2 = cv2.imread('data/bear.jpg')
filename = 'trump_bear'
#assert im1.shape == im2.shape, "Images does not have the same shapes."

# Get points
points_1, points_2 = getPointCorrespondences(im1, im2)  # [[x y] [x y]]
print("points_1:", points_1)
print("points_2:", points_2)

# morph with delaunay triangulation
morph = delaunay_triangulation(im1, im2, points_1, points_2, 0)

i = 0
step_size = 0.1
count = 0
while True:
    cv2.imshow("Output", morph)

    k = cv2.waitKey(20) & 0xFF
    # Press n button for next morph
    if k == ord('n'):
        i += step_size
        if i > 1:
            i = 1

        morph = delaunay_triangulation(im1, im2, points_1, points_2, i)
        print(i)


    if k == ord('p'):
    # Press p button for previous morph
        i -= step_size
        if i < 0:
            i = 0
        morph = delaunay_triangulation(im1, im2, points_1, points_2, i)
        print(i)

    if k == ord('s'):
        cv2.imwrite(filename + '_morph{}.jpg'.format(count), morph)
        count += 1
        print(filename + '_morph{}.jpg'.format(i))

cv2.destroyAllWindows()
