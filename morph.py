#!/usr/bin/python

import cv2
import numpy as np
from pointCorrespondences import automatic_point_correspondences
from numpy.linalg import inv


def transform_points(points, H_inv):
    z = np.ones(shape=(1, len(points)))
    points = np.concatenate((points, z.T), axis=1)
    points_trans = np.matmul(H_inv, points.T)


    print(points_trans.shape)
    return np.array(points_trans.T).astype(np.uint8)


def mapDelaunay(triangles_A, points_A, points_B, points_C):
    triangles_B = []
    triangles_C = []
    #print(triangles_A)
    #print(points)
    points_A = np.uint8(points_A)
    triangles_A = np.uint8(triangles_A)
    for tri in triangles_A:
        tri_B = []
        tri_C = []
        for i in range(0, 6, 2):
            #idx = points_A.index([int(tri[0+i]), int(tri[1+i])])
            #print(tri[i])
            index = np.where(points_A==tri[i])

            #index = np.where(np.isclose(points_A, tri[i], 0.1))
            #print(index)
            for idx in index[0]:
                if points_A[idx][1] == tri[1+i]:
                    tri_B.extend(points_B[idx])
                    tri_C.extend(points_C[idx])


        # if len(tri_B) < 6:
        #     print(tri_B)
        #     print('B: ', tri_B)
        #     print(index)
        # if len(tri_C) < 6:
        #     print('C: ', tri_C)
        #     print(index)
        triangles_B.append(tri_B)

        triangles_C.append(tri_C)
        #print(len(triangles_B), len(triangles_A), len(triangles_C))

    return triangles_B, triangles_C


def applyAffineTransform(im_src, t_scr, dest_scr, size):
    """
    Applies affine transoformation
    :param im_src:
    :param t_scr:
    :param dest_scr:
    :param size:
    :return:
    """
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(t_scr), np.float32(dest_scr))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(im_src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

def remove_points(points_1, points_2, remove_idx=None):
    if remove_idx is None:
        remove_idx = [1, 3, 5, 7, 9, 11, 13, 15, 50, 60, 61, 62, 68, 63, 64, 66, 54, 65, 56, 33, 35]


    points_1 = np.delete(points_1, remove_idx, 0)
    points_2 = np.delete(points_2, remove_idx, 0)

    return points_1, points_2

def delaunay_triangulation(image1, image2, points_1, points_2, morphshape, removepoints=True, alph=0.5):
    """
    Performs morphing of two images with delaunay triangulation. Alph indicates how much morphed image looks like image 2
    :param image1: image of a face
    :param image2: image of another face
    :param points_1: list of facial feature points of image1
    :param points_2: list of facial feature points of image2, corresponding to points in image1
    :param alph: between 0 and 1. Indicates likeness to one image
    :return: morphed image
    """
    # Remove some points to avoid overcrowding
    if removepoints:
        points_1, points_2 = remove_points(points_1, points_2)

    # Get intermediate points for generated image
    points_k = (1-alph)*points_1 + alph*points_2

    # Perform delaunay triangulation with subdiv on one pointset
    size = image1.shape
    rect = (0, 0, size[1], size[0])

    # Create instances of Subdiv2D
    subdiv1 = cv2.Subdiv2D(rect)

    # Insert points in the subdiv
    insert_points(subdiv1, points_1)

    # Get Delaunay triangles from the subdiv. Get the corresponding triangles for the two other "images"
    triangles_1 = subdiv1.getTriangleList()
    triangles_2, triangles_k = mapDelaunay(triangles_1, points_1, points_2, points_k)

    # initiate morphed image placeholder
    morph_im = np.zeros(shape=np.array([morphshape[0], morphshape[1], 3]))

    for i in range(0, len(triangles_k)):
        # Find bounding rectangle
        t1 = triangles_1[i]
        t2 = triangles_2[i]
        tk = triangles_k[i]

        show = False
        while(show):
            array1 = np.array([(t1[0], t1[1]), (t1[2], t1[3]), (t1[4], t1[5])])
            cv2.drawContours(image1, [array1.astype(int)], 0, (0, 255, 0), -1)
            cv2.imshow('window', image1)

            array2 = np.array([(t2[0], t2[1]), (t2[2], t2[3]), (t2[4], t2[5])])
            cv2.drawContours(image2, [array2.astype(int)], 0, (0, 255, 0), -1)
            cv2.imshow('window2', image2)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                print('array 1: ', array1)
                print('array 2: ', array2)
                print('----------------------------------------')
                show = False

        # Bounding rectangles created
        r1 = cv2.boundingRect(np.float32([(t1[0], t1[1]),
                                          (t1[2], t1[3]),
                                          (t1[4], t1[5])]))
        r2 = cv2.boundingRect(np.float32([(t2[0], t2[1]),
                                          (t2[2], t2[3]),
                                          (t2[4], t2[5])]))
        rk = cv2.boundingRect(np.float32([(tk[0], tk[1]),
                                          (tk[2], tk[3]),
                                          (tk[4], tk[5])]))

        # If show == True this will draw rectangles on the images in the order they appear
        # Useful if one suspects the triangle correspondence is not working
        show = False
        while(show):
            cv2.rectangle(image1, (r1[0], r1[1]), (r1[0]+r1[2], r1[1]+r1[3]), (0,255,0),3)
            cv2.imshow('window', image1)

            cv2.rectangle(image2, (r2[0], r2[1]), (r2[0]+r2[2], r2[1]+r2[3]), (0,255,0),3)
            cv2.imshow('window2', image2)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                show = False


        # Get triangle position within the rectangles
        t1Rect = []
        t2Rect = []
        tkRect = []

        for j in range(0, 3):
            tkRect.append(((tk[0+2*j] - rk[0]), (tk[1+2*j] - rk[1])))
            t1Rect.append(((t1[0+2*j] - r1[0]), (t1[1+2*j] - r1[1])))
            t2Rect.append(((t2[0+2*j] - r2[0]), (t2[1+2*j] - r2[1])))

        # Apply warp to the rectangular patches
        img1Rect = image1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (rk[2], rk[3])
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tkRect, size)
        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tkRect, size)


        # Fill in the morphed image
        imgRect = (1.0 - alph) * np.float32(warpImage1) + alph * warpImage2


        m = morph_im[rk[1]:rk[1] + rk[3], rk[0]:rk[0] + rk[2]]
        m_shape = m.shape
        rect_shape = imgRect.shape
        if m_shape != rect_shape:
            imgRect = cv2.resize(imgRect, (m_shape[1], m_shape[0]), interpolation=cv2.INTER_AREA)
            print(imgRect.shape)

        # Create mask and fill the triangle
        mask = np.zeros(imgRect.shape, dtype=np.float32)

        # Fill the triangle in the mask
        cv2.fillConvexPoly(mask, np.int32(tkRect), (1.0, 1.0, 1.0), 16, 0)


        # Copy triangular region of the rectangular patch to the output image
        morph_im[rk[1]:rk[1] + rk[3], rk[0]:rk[0] + rk[2]] = morph_im[rk[1]:rk[1] + rk[3], rk[0]:rk[0] + rk[2]] * (1 - mask) \
                                                             + imgRect * mask


        show = False
        while(show):
            cv2.imshow('window2', morph_im)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                show = False

    return morph_im.astype(np.uint8)


def insert_points(subdiv, p_list):
    """
    Insert a list of points in a subdiv
    :param subdiv: instance of subdiv
    :param p_list: list of points
    """
    for i in p_list:
        subdiv.insert(tuple(i))


if __name__ == '__main__':
    image1 = cv2.imread('data/einstein1.jpg')
    image2 = cv2.imread('data/einstein3.jpg')
    filename = 'einstein'
    print(image1.shape)

    # Get points with dlib facial feature point detector
    points_1, points_2 = automatic_point_correspondences(image1, image2)

    # morph with delaunay triangulation
    morph = delaunay_triangulation(image1, image2, points_1, points_2, 0)

    i = 0
    step_size = 0.2
    count = 0
    while True:
        cv2.imshow("Output", morph)
        k = cv2.waitKey(20) & 0xFF

        # N: next worph
        if k == ord('n'):
            i += step_size
            if i > 1:
                i = 1
            morph = delaunay_triangulation(image1, image2, points_1, points_2, i)
            print("i: {:.2f}".format(i))

        # P: previous morph
        if k == ord('p'):
            i -= step_size
            if i < 0:
                i = 0
            morph = delaunay_triangulation(image1, image2, points_1, points_2, i)
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
