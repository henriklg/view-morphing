import numpy as np
import cv2
from matplotlib import pyplot as plt
from findfundmat import findfundmat
from pointCorrespondences import automatic_point_correspondences

def drawlines(img1, img2, lines, pts1, pts2, color):
    """
    Draw epipolar line on image

    Args:
    imgLeft - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
    """
    r, c = img1.shape

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2, co in zip(lines, pts1, pts2, color):
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), co, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, co, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, co, -1)
    return img1, img2

def find_epilines(imgLeft, imgRight, ptsLeft, ptsRight, F):
    """
    finds the epipolar lines in two images given a set of point-correspondences

    Args:
        imgLeft : left image
        imgRight: right image
        ptsLeft : points in left image
        ptsRight: points in right image
        F       : Fundamental matrix
    Returns:
        new_img1
        new_img2
    """
    color = []
    for i in range(ptsLeft.shape[0]):
        color.append(tuple(np.random.randint(0, 255, 3).tolist()))
    print(color)

    # Find epilines corresponding to points in right image (right image)
    linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2), 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    # Draw its lines on left image
    img5, img6 = drawlines(imgLeft, imgRight, linesLeft, ptsLeft, ptsRight, color)

    # Find epilines corresponding to points in left image (left image)
    linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
    linesRight = linesRight.reshape(-1, 3)
    # Draw its lines on right image
    img3, img4 = drawlines(imgRight, imgLeft, linesRight, ptsRight, ptsLeft, color)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

if __name__ == '__main__':

    imgLeft = cv2.imread('einstein2.jpg')
    imgRight = cv2.imread('einstein1.jpg')
    x_1, y_1, x_2, y_2 = automatic_point_correspondences(imgLeft, imgRight)
    ptsLeft = np.concatenate((x_1, y_1), axis=1).reshape(x_1.shape[0],2)
    ptsRight = np.concatenate((x_2, y_2), axis=1).reshape(x_1.shape[0],2)

    #choose 3 points
    ptsLeft = np.array([ptsLeft[0,:], ptsLeft[30,:], ptsLeft[60,:]])
    ptsRight = np.array([ptsRight[0,:], ptsRight[30,:], ptsRight[60,:]])

    F = findfundmat(imgLeft, imgRight)
    find_epilines(imgLeft[:,:,0], imgRight[:,:,0], ptsLeft, ptsRight, F)
