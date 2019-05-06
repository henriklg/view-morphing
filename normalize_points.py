import numpy as np

def normalize_points(x, y):
    """
    function computing normalization (translation and scaling) of the coordinates of the matched points

    :param x; (n x 1 ndarray) x coordinates of points
    :param y: (n x 1 ndarray) y coordiantes of points
    :return:
    """
    num_points = len(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    shifted_x = x - mean_x
    shifted_y = y - mean_y

    sf = np.sqrt(2) / np.mean( np.sqrt(shifted_x**2 + shifted_y**2) )
    # 3x3 transformation matrix
    t_norm = np.array([[sf, 0, -sf*mean_x], [0, sf, -sf*mean_y], [0, 0, 1]])

    o = np.ones((num_points,1))

    xy = np.concatenate((x, y, o), axis=0).reshape(-1, 4)

    # (n x 3) normalized homogeneous coordinates of points
    xy_normalized = np.transpose(np.matmul(t_norm, xy))

    return xy_normalized, t_norm