
import numpy as np
from getcoords import getcoords
from normalize_points import normalize_points

def findfundmat(image1, image2):
    """
    calculates a fundamental matrix for the two specified images

    :param image1:
    :param image2:
    :return:
    """
    u, v, up, vp = getcoords(image1, image2)

    F = np.zeros((3, 3))

    # get the number of data points
    num_points = len(u)
    o = np.ones((num_points, 1))

    uv, t1 = normalize_points(u, v)
    upvp, t2 = normalize_points(up, vp)

    u = upvp[:, 0]
    v = upvp[:, 1]
    up = uv[:, 0]
    vp = uv[:, 1]


    # compute A - the equation matrix
    mul = lambda a, b : np.multiply(a, b)
    a = np.concatenate((mul(u, up), mul(u, vp), u, mul(v, up), mul(v, vp), v, up, vp, o[:, 0])).reshape((4,-1), order='F')

    # the singular value decomposition SVD
    U, D, V = np.linalg.svd(a)

    # extract column of the smallest singular value - the last column
    smallest = V[:, 8]
    #print (smallest)
    F = smallest.reshape(3, 3)

    # enforce singularity constraint (must be singular and of rank 2)
    U, D, V = np.linalg.svd(F)
    r = D[1]
    s = D[2]

    F = U*np.diag([r, s, 0])*V.T
    F = t2.T * F * t1

    return F


image1 = 'test1'
image2 = 'test2'
findfundmat(image1, image2)
