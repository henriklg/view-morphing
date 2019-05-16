import cv2


def get_coords(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = x, y
        print(point)

def getRectangle(im):
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