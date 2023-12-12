import numpy as np

def estimate_pose(x, X):
    """
    Computes the pose matrix (camera matrix) P given 2D and 3D points.

    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]

    Returns:
        P: Pose matrix (camera matrix) with shape [3, 4]
    """

    N = x.shape[1]

    x_new = np.vstack((x, np.ones((1, N))))
    X_new = np.vstack((X, np.ones((1, N))))

    A = np.zeros((2 * N, 12))
    for i in range(N):
        A[2 * i, :4] = X_new[:, i]
        A[2 * i + 1, 4:8] = X_new[:, i]
        A[2 * i, 8:] = -x_new[0, i] * X_new[:, i]
        A[2 * i + 1, 8:] = -x_new[1, i] * X_new[:, i]

    _, _, V = np.linalg.svd(A)

    P = V[-1, :].reshape(3, 4)

    return P
