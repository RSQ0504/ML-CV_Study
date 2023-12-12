import numpy as np

def dist(im1,im2,y,x,windowSize,d):
    w = int((windowSize - 1)/2)
    result = 0
    for i in range(-w,w):
        for j in range(-w,w):
            result += (im1[y+i,x+j] - im2[y+i,x+j-d]) ** 2
    return result

def dispM_yx(im1,im2,y,x,maxDisp,windowSize):
    min = np.inf
    for d in range(maxDisp + 1):
        temp = dist(im1,im2,y,x,windowSize,d)
        if min > temp:
            min = temp
            result = d
    return result

def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    dispM = np.zeros_like(im1, dtype=float)
    
    for y in range(im1.shape[0]):
        for x in range(im1.shape[1]):
            dispM[y,x] = dispM_yx(im1,im2,y,x,maxDisp,windowSize)
    
    return dispM

