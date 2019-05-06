import numpy as np
import cv2
import dlib
from imutils import face_utils
from time import sleep


def getPointCorrespondences(file_1, file_2):
    global point
    point = (-1, -1)
    point_click = (-1, -1)
    point_list = []
    im1 = cv2.imread(file_1)
    height, width, _ = im1.shape
    print(height, width)
    im2 = cv2.imread(file_2)

    numpy_horizontal = np.hstack((im1, im2))
    numpy_orig = numpy_horizontal.copy()


    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Images', get_coords)
    #cv2.setMouseCallback('Image2', get_coords)

    while(1):
        cv2.imshow('Images', numpy_horizontal)
        #cv2.imshow('Image2', im2)
        k = cv2.waitKey(20) & 0xFF

        if point_click != point:
            cv2.circle(numpy_horizontal, point, 2, (0, 0, 255), 2)
            point_click = point
            point_list.append(point_click)
            print(point_list)
        if k == 27:
            break
        elif k == ord('s'):
            point_correspondences = []
            count = 0
            add_point = []
            for p in point_list:
                if count % 2 == 0:
                    add_point.append(p)
                    count += 1
                else:
                    x = p[0] - width
                    y = p[1]
                    add_point.append((x,y))
                    point_correspondences.append(add_point)
                    add_point = []
                    count += 1
            print(point_correspondences)



        elif k == ord('a'):
            print(posList1, posList2)

        elif k == ord('s'):
            if len(posList1) == len(posList2):
                return posList1, posList2
            else:
                print('ERROR')
        elif k == ord('n'):
            posList1 = []
            posList2 = []
            numpy_horizontal = numpy_orig.copy()
            cv2.imshow('Images', numpy_horizontal)

    cv2.destroyAllWindows()
    return posList


def get_coords(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = x, y
        print(point)

def return_coords(x, y):
    return x, y


def automatic_point_correspondences_imshow(file_1, file_2):
    # Load pre-trained facial features model
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Load images and convert to grayscale
    im1 = cv2.imread(file_1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im = im1.copy()
    im2 = cv2.imread(file_2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect faces with the detector
    rects_1 = detector(im1, 0)
    rects_2 = detector(im2, 0)

    detect1 = True
    detect2 = True
    # For each detected face in each image, draw the points
    while(True):
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

def automatic_point_correspondences(file_1, file_2):
    # Load pre-trained facial features model
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Load images and convert to grayscale
    im1 = cv2.imread(file_1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread(file_2)
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

    return shape_1[:, 0], shape_1[:, 1], shape_2[:, 0], shape_2[:, 1]


if __name__ == '__main__':
    #posList = []
    x_1, y_1, x_2, y_2 = automatic_point_correspondences('einstein1.jpg', 'einstein3.jpg')
    #print(posList)
