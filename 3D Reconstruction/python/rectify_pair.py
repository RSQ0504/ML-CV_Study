import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE

    M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

    c1 = np.dot(-np.linalg.inv(np.dot(K1, R1)), (np.dot(K1, t1)))
    c2 = np.dot(-np.linalg.inv(np.dot(K2, R2)), (np.dot(K2, t2)))

    r1 = (np.abs(c1 - c2) / np.linalg.norm(c1 - c2)).reshape((-1))

    r21 = np.cross(R1[2, :], r1)
    r21 = r21 / np.linalg.norm(r21)
    r31 = np.cross(r1, r21)
    r31 = r31 / np.linalg.norm(r31)
    R1n = np.vstack((r1, r21, r31))

    r22 = np.cross(R2[2, :], r1)
    r32 = np.cross(r1, r22)
    R2n = np.vstack((r1, r22, r32))

    K1n = K2
    K2n = K2

    t1n = -np.dot(R1n, c1)
    t2n = -np.dot(R2n, c2)

    M1 = np.dot(np.dot(K1n, R1n), np.linalg.inv(np.dot(K1, R1)))
    M2 = np.dot(np.dot(K2n, R2n), np.linalg.inv(np.dot(K2, R2)))

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

