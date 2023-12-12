import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    depthM = np.zeros_like(dispM, dtype=float)
    c1 = np.dot(-np.linalg.inv(np.dot(K1, R1)), np.dot(K1, t1))
    c2 = np.dot(-np.linalg.inv(np.dot(K2, R2)), np.dot(K2, t2))

    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]

    nonzero_indices = dispM != 0
    depthM[nonzero_indices] = b * f / dispM[nonzero_indices]
    return depthM

