import numpy as np


def to_camera(X, K, RT, d=None, ret_homogeneous=False):
    """ 
    X must be a matriz 3xN with x, y and z coordinates on each row 
    """
    if X.shape[0] != 3:
        raise Exception("'X' first dimension must be equals 3.")

    X_ = np.vstack([X, np.ones(X.shape[1])])
    x = np.asarray(np.matmul(RT, X_))
    x[0:2, :] = x[0:2, :] / x[2, :]
    x[2, :] = 1.0

    if d is not None:
        u, v = x[0, :], x[1, :]
        r2 = u * u + v * v
        radial_factor = 1 + d[0] * r2 + d[1] * (r2**2) + d[4] * (r2**3)
        u_tangential_factor = 2 * d[2] * u * v + d[3] * (r2 + 2 * u * u)
        v_tangential_factor = 2 * d[3] * u * v + d[2] * (r2 + 2 * v * v)
        x[0, :] = u * radial_factor + u_tangential_factor
        x[1, :] = v * radial_factor + v_tangential_factor

    x = np.matmul(K, x[0:3, :])
    if ret_homogeneous:
        return x[0:3, :]
    else:
        return x[0:2, :]


def validate_resolution(joints, width, height):
    if joints.shape[0] != 2:
        raise Exception("'joints' array first shape must be equals 2.")
    return np.logical_not(np.logical_or(joints[0, :] > width, joints[1, :] > height))
