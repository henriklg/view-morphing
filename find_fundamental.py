import numpy as np
from pointCorrespondences import automatic_point_correspondences


def normalize_points(x, y):
    """
    function computing normalization (translation and scaling) of the coordinates of the matched points

    :param x; (n x 1 ndarray) x coordinates of points
    :param y: (n x 1 ndarray) y coordinates of points
    :return: normalized points and 3x3 transformation matrix
    """
    assert len(x) == len(y), "x, y points are not of same length."
    num_points = len(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    shifted_x = x - mean_x
    shifted_y = y - mean_y

    sf = np.sqrt(2) / np.mean( np.sqrt(shifted_x**2 + shifted_y**2) )
    # 3x3 transformation matrix
    t_norm = np.array([[sf, 0, -sf*mean_x], [0, sf, -sf*mean_y], [0, 0, 1]])

    ones = np.ones((num_points, 1))
    xy = np.concatenate((x, y, ones), axis=1)

    # (n x 3) normalized homogeneous coordinates of points
    xy_normalized = np.dot(t_norm, xy.T)

    return xy_normalized.T, t_norm



def fundamental_matrix(image1, image2):
    """
    calculates a fundamental matrix for the two specified images

    :param image1:
    :param image2:
    :return: fundamental matrix F (3 x 3)
    """
    [x1, y1, x2, y2] = automatic_point_correspondences(image1, image2)

    # get the number of data points
    num_points = len(x1)
    o = np.ones((num_points, 1))

    x1y1, t1 = normalize_points(x1, y1)
    x2y2, t2 = normalize_points(x2, y2)

    x1 = x2y2[:, 0]
    y1 = x2y2[:, 1]
    x2 = x1y1[:, 0]
    y2 = x1y1[:, 1]

    # compute A - the equation matrix
    mul = lambda a, b : np.multiply(a, b)
    a = np.concatenate((mul(x1, x2), mul(x1, y2), x1, mul(y1, x2), mul(y1, y2), y1, x2, y2, o[:, 0])).reshape((num_points,-1), order='F')

    # the singular value decomposition SVD
    U, D, V = np.linalg.svd(a)

    # extract column of the smallest singular value - the last column
    smallest = V[8, :].T
    F = smallest.reshape(3, 3)

    # enforce singularity constraint (must be singular and of rank 2)
    U, D, V = np.linalg.svd(F)
    r = D[0]
    s = D[1]

    F = np.dot(U, np.diag([r, s, 0])).dot(V)
    F = t2.T.dot(F).dot(t1)

    return F


if __name__ == '__main__':
    image1 = 'data/einstein1.jpg'
    image2 = 'data/einstein3.jpg'

    F = fundamental_matrix(image1, image2)
    print(F)