import numpy as np

def essentialMatrix(F, K1, K2):
    """
    Args:
        F:  Fundamental Matrix
        K1: Camera Matrix 1
        K2: Camera Matrix 2   
    Returns:
        E:  Essential Matrix  
    """
    if F.shape != (3, 3):
        raise ValueError("Invalid shape for the fundamental matrix F")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("Invalid shape for camera matrices K1 or K2")

    E = np.dot(K2.T, np.dot(F, K1))

    return E
