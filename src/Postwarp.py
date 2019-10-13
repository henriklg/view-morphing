import cv2
import numpy as np


def get_coords(event, x, y, flags, param):
    """
    Gets coordinates of mouse in the event of left mouse button down
    """
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = x, y
        print(point)

def getRectangle(im):
    """
    Creates rectangles from selected points on image
    """
    global point
    point = (-1, -1)
    point_click = (-1, -1)
    m_point_list = []
    im_show = im.copy()


    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', get_coords)

    while True:
        cv2.imshow('Image', im_show)
        k = cv2.waitKey(20) & 0xFF

        if point_click != point:
            cv2.circle(im_show, point, 2, (0, 255, 0), 2)
            point_click = point
            m_point_list.append(point_click)

            if len(m_point_list) == 2:
                cv2.rectangle(im_show, m_point_list[0], m_point_list[1], (255, 0, 255), 2)


        # Restart
        elif k == ord('r'):
            im_show = im.copy()
            point = (-1, -1)
            point_click = (-1, -1)
            m_point_list = []

        # s: return points
        elif k == ord('s'):
            break
    cv2.destroyAllWindows()
    return m_point_list

def getPoints(im):
    """
    Draws and returns selected points on image
    """
    global point
    point = (-1, -1)
    point_click = (-1, -1)
    m_point_list = []
    im_show = im.copy()
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', get_coords)

    while True:
        cv2.imshow('Image', im_show)
        k = cv2.waitKey(20) & 0xFF

        if point_click != point:
            cv2.circle(im_show, point, 1, (255, 0, 255), 2)
            point_click = point
            m_point_list.append(point_click)

        # Restart
        elif k == ord('r'):
            im_show = im.copy()
            point = (-1, -1)
            point_click = (-1, -1)
            m_point_list = []

        elif k == ord('l'):
            cv2.imwrite('point_einstein.jpg', im_show)

        # s: return points
        elif k == ord('s'):
            break
    cv2.destroyAllWindows()
    return m_point_list

def getLines(im):
    """
    Draws lines between selected points on image. Returns points
    """
    global point
    point = (-1, -1)
    point_click = (-1, -1)
    m_point_list = []
    im_show = im.copy()

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', get_coords)

    while True:
        cv2.imshow('Image', im_show)
        k = cv2.waitKey(20) & 0xFF

        if point_click != point:
            cv2.circle(im_show, point, 2, (0, 255, 0), 2)
            point_click = point
            m_point_list.append(point_click)

            if len(m_point_list) >= 2:
                cv2.line(im_show, m_point_list[-1], m_point_list[-2], (255, 0, 255), 2)

            if len(m_point_list) == 4:
                cv2.line(im_show, m_point_list[-1], m_point_list[0], (255, 0, 255), 2)



        # Restart
        elif k == ord('r'):
            im_show = im.copy()
            point = (-1, -1)
            point_click = (-1, -1)
            m_point_list = []

        # s: return points
        elif k == ord('s'):
            break
    cv2.destroyAllWindows()
    return m_point_list

def homography(m_points, p_points):
    m_points = np.asarray(m_points, dtype=np.uint8)
    p_points = np.asarray(p_points, dtype=np.uint8)

    pts_src = np.array([[m_points[0, 0], m_points[0, 1]], [m_points[2, 0], m_points[2, 1]],
                        [m_points[3, 0], m_points[3, 1]], [m_points[1, 0], m_points[1, 1]]])

    pts_dest = np.array([[p_points[0, 0], p_points[0, 1]], [p_points[1, 0], p_points[1, 1]],
                         [p_points[0, 0], p_points[1, 1]], [p_points[1, 0], p_points[0, 1]]])

    H_s, _ = cv2.findHomography(pts_src, pts_dest)
    return H_s

def homography_points(m_points, p_points):
    m_points = np.asarray(m_points, dtype=np.uint8)
    p_points = np.asarray(p_points, dtype=np.uint8)

    pts_src = []
    pts_dest = []
    for i in range(0, len(m_points)-1):
        pts_src.append([m_points[i, 0], m_points[i, 1]])
        pts_dest.append([p_points[i, 0], p_points[i, 1]])

    pts_src = np.asarray(pts_src, dtype=np.uint8)
    pts_dest = np.asarray(pts_dest, dtype=np.uint8)

    H_s, _ = cv2.findHomography(pts_src, pts_dest)
    return H_s

