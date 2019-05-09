import numpy as np
from normalize_points import normalize_points
from pointCorrespondences import automatic_point_correspondences

def findfundmat(image1, image2):
    """
    calculates a fundamental matrix for the two specified images

    :param image1:
    :param image2:
    :return:
    """
    [x1, y1, x2, y2] = automatic_point_correspondences(image1, image2)
    #F = np.zeros((3, 3))

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
    image1 = 'einstein1.jpg'
    image2 = 'einstein3.jpg'

    fundamental_matrix = findfundmat(image1, image2)
    print(fundamental_matrix)