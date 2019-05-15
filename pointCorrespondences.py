import numpy as np
import cv2
import dlib
from imutils import face_utils
from time import sleep


def getPointCorrespondences(im1, im2):
    global point
    point = (-1, -1)
    point_click = (-1, -1)
    point_list = []
    height, width, _ = im1.shape

    numpy_horizontal = np.hstack((im1, im2))
    numpy_orig = numpy_horizontal.copy()

    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Images', get_coords)

    while True:
        cv2.imshow('Images', numpy_horizontal)
        k = cv2.waitKey(20) & 0xFF

        if point_click != point:
            cv2.circle(numpy_horizontal, point, 2, (0, 0, 255), 2)
            point_click = point
            point_list.append(point_click)

        if k == 27:
            break

        # S: return list of points
        elif k == ord('s'):
            count = countx = county = 0
            pointx = pointy = np.zeros(( (len(point_list)+1)//2, 2))  # [num_points, 2]
            for p in point_list:
                # image 1
                if count % 2 == 0:
                    pointx[countx] = [p[0], p[1]]
                    countx += 1
                # image 2
                else:
                    pointy[county] = [p[0]-width, p[1]]
                    county += 1
                count += 1
            return pointx, pointy

        # N: new list of points
        elif k == ord('n'):
            point_list = []
            numpy_horizontal = numpy_orig.copy()
            cv2.imshow('Images', numpy_horizontal)

    cv2.destroyAllWindows()
    return point_list


def get_coords(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = x, y
        print(point)


def automatic_point_correspondences_imshow(im1, im2):
    # Load pre-trained facial features model
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Load images and convert to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im = im1.copy()
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect faces with the detector
    rects_1 = detector(im1, 0)
    rects_2 = detector(im2, 0)

    detect1 = True
    detect2 = True
    # For each detected face in each image, draw the points
    while (True):
        k = cv2.waitKey(20) & 0xFF
        # Image 1:
        if detect1:
            for (i, rect) in enumerate(rects_1):
                # Make the prediction and transfom it to numpy array
                shape = predictor(im1, rect)
                shape_1 = face_utils.shape_to_np(shape)

            detect1 = False

        if detect2:
            for (i, rect) in enumerate(rects_2):
                # Make the prediction and transfom it to numpy array
                shape = predictor(im2, rect)
                shape_2 = face_utils.shape_to_np(shape)

            detect2 = False

        if len(shape_1) == len(shape_2):
            for i in range(len(shape_1)-1):
                cv2.circle(im1, tuple(shape_1[i]), 2, (0, 255, 0), -1)
                cv2.imshow('Image 1', im1)
                cv2.circle(im2, tuple(shape_2[i]), 2, (255, 0, 0), -1)
                cv2.imshow('Image 2', im2)


        if k == ord('s'):
            cv2.destroyAllWindows()
            break

    im1 = im
    cv2.circle(im1, tuple(shape_1[0]), 2, (0, 255, 0), -1)
    cv2.circle(im1, tuple(shape_1[8]), 2, (0, 255, 0), -1)
    cv2.circle(im1, tuple(shape_1[16]), 2, (0, 255, 0), -1)
    cv2.circle(im1, tuple(shape_1[19]), 2, (0, 255, 0), -1)
    cv2.circle(im1, tuple(shape_1[0]), 2, (0, 255, 0), -1)
    cv2.imshow('Image', im1)

    k = cv2.waitKey(20) & 0xFF


    return shape_1, shape_2


def automatic_point_correspondences(im1, im2):
    # Load pre-trained facial features model
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Load images and convert to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect faces with the detector
    rects_1 = detector(im1, 0)
    rects_2 = detector(im2, 0)

    # Image 1
    for (i, rect) in enumerate(rects_1):
        # Make the prediction and transfom it to numpy array
        shape = predictor(im1, rect)
        shape_1 = face_utils.shape_to_np(shape)

    # Image 2
    for (i, rect) in enumerate(rects_2):
        # Make the prediction and transfom it to numpy array
        shape = predictor(im2, rect)
        shape_2 = face_utils.shape_to_np(shape)

    # some magic to make the coordinates compatible with other code
    x1 = shape_1[:, 0][np.newaxis].T
    y1 = shape_1[:, 1][np.newaxis].T
    x2 = shape_2[:, 0][np.newaxis].T
    y2 = shape_2[:, 1][np.newaxis].T

    return x1, y1, x2, y2


if __name__ == '__main__':
    im1 = cv2.imread('einstein1.jpg')
    im2 = cv2.imread('einstein3.jpg')
    x_1, y_1, x_2, y_2 = automatic_point_correspondences(im1, im2)

    str = '['
    for point in y_2:
        str = str + '{}; '.format(point[0])
    str = str + ']'
    print(str)

    #print((x_1)[np.newaxis].T)
