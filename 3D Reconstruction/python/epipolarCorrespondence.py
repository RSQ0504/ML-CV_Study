import numpy as np
import cv2

def normalized_cross_correlation(window1, window2):
    """
    Calculate normalized cross-correlation between two windows.
    """
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)
    numerator = np.sum((window1 - mean1) * (window2 - mean2))
    denominator = np.sqrt(np.sum((window1 - mean1)**2) * np.sum((window2 - mean2)**2))
    return numerator / denominator if denominator != 0 else 0

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    window_size = 15
    pts2 = np.zeros_like(pts1)
    
    for i in range(pts1.shape[0]):
        point = np.array([pts1[i, 0], pts1[i, 1], 1])
        #print(point)
        epipolar_line = np.dot(F, point)
        best_score = -float('inf')
        best_point = np.zeros(2, dtype=int)
        window = im1[int(pts1[i, 1]) - int(window_size / 2): int(pts1[i, 1]) + int(window_size / 2 + 1),
                    int(pts1[i, 0]) - int(window_size / 2):int(pts1[i, 0]) + int(window_size / 2 + 1)]

        for j in np.arange(0,im2.shape[0],0.1):
            x_candidate = float(((-epipolar_line[1] * j - epipolar_line[2]) / epipolar_line[0]))
            if x_candidate <= 0 or x_candidate >im2.shape[1]:
                continue
            temp_window = im2[int(j) - int(window_size / 2):int(j) + int(window_size / 2) + 1, 
                                int(x_candidate) - int(window_size / 2):int(x_candidate) + int(window_size / 2) + 1]
            #print(window.shape,temp_window.shape)
            if window.shape == temp_window.shape:
                score = normalized_cross_correlation(window, temp_window)
                #print(ssd_score)
                if score > best_score:
                    best_score = score
                    best_point = np.array([x_candidate, j])
                    #print(best_point)
        pts2[i, :] = best_point
        #print(point,best_point)
    return pts2

if __name__ == "__main__":
    from eightpoint import eightpoint
    import cv2
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join('..', 'data', 'im1.png')
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))

    im1 = cv2.imread(image_path)

    relative_path = os.path.join('..', 'data', 'im2.png')
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))
    im2 = cv2.imread(image_path)

    correspondence = np.load('../data/someCorresp.npy', allow_pickle=True).item()

    pts1 = correspondence['pts1']
    pts2 = correspondence['pts2']

    M = max(im1.shape[0],im1.shape[1])
    F = eightpoint(pts1, pts2, M)

    pts2 = epipolarCorrespondence(im1, im2, F, pts1)
    print(pts2)