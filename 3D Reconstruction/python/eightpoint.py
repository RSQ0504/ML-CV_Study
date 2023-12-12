import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    # Normalize the points
    pts1 = pts1 / M
    pts2 = pts2 / M

    A = np.column_stack((pts2[:, 0] * pts1[:, 0], pts2[:, 0] * pts1[:, 1], pts2[:, 0], pts2[:, 1] * pts1[:, 0],
                         pts2[:, 1] * pts1[:, 1], pts2[:, 1], pts1[:, 0], pts1[:, 1], np.ones(pts1.shape[0])))

    _, _, V = np.linalg.svd(A)

    F = V[-1, :].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    # print(S)
    F = np.dot(U, np.dot(np.diag(S), V))

    F = refineF(F, pts1, pts2)
    
    # 齐次坐标, 深度信息
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = np.dot(T.T, np.dot(F, T))
    
    return F

if __name__ == "__main__":
    from displayEpipolarF import displayEpipolarF
    import cv2
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join('..', 'data', 'im1.png')
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))

    I1 = cv2.imread(image_path)

    relative_path = os.path.join('..', 'data', 'im2.png')
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))
    I2 = cv2.imread(image_path)

    correspondence = np.load('../data/someCorresp.npy', allow_pickle=True).item()

    pts1 = correspondence['pts1']
    pts2 = correspondence['pts2']

    M = max(I1.shape[0],I1.shape[1])

    F = eightpoint(pts1, pts2, M)
    print(F)
    displayEpipolarF(I1,I2,F)