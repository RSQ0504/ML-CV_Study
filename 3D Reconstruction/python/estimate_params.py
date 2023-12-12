import numpy as np
from scipy.linalg import svd, qr

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    K, R, t = None, None, None
    _, _, V = np.linalg.svd(P)
    c = V[-1, :-1] / V[-1, -1]


    reverse = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    P_r = np.dot(reverse, P[:, :3])
    P_r = P_r.T
    Q, R = qr(P_r)

    Q = np.dot(reverse, Q.T) # 𝑄  is orthogonal
    R = np.dot(np.dot(reverse, R.T), reverse) # upper triangular 

    K = R
    R = Q
    

    D = np.diag(np.sign(np.diag(K)))
    K = np.dot(K,D)
    R = np.dot(D,R)
    if (abs(np.linalg.det(R) + 1) < 1e-4):
        R = -R
    K = K / K[-1, -1]
    
    t = -np.dot(R, c.reshape(-1, 1))

    return K, R, t

