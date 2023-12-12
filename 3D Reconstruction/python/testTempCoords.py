import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI

# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
M = max(img1.shape[0],img1.shape[1])


# write your code here
R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = np.eye(3), np.zeros((3, 1))

F = eightpoint(pts1, pts2, M)
print(F)

pts = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = epipolarCorrespondence(img1, img2, F, pts1)
intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
k1 = intrinsics['K1']
k2 = intrinsics['K2']

E = essentialMatrix(F, k1, k2)
print(E)

P1 = np.dot(k1,np.hstack((np.eye(3), np.zeros((3, 1)))))

P2_possible_e = camera2(E)
best = -1

for i in range(P2_possible_e.shape[-1]):
    temp_p2_e = P2_possible_e[:,:,i]
    temp_P2 = np.dot(k2,temp_p2_e)
    points = triangulate(P1, pts1, temp_P2, pts2)
    front_count = 0
    for point in points:
        if point[-1]>0:
            front_count += 1
    if front_count > best:
        best = front_count
        P2 = temp_P2
        R2 = temp_p2_e[:,:-1]
        t2 = temp_p2_e[:,-1].reshape(-1,1)
        result_points = points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(result_points[:, 0], result_points[:, 1], result_points[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
points = triangulate(P1, pts1, P2, pts2)
points = np.hstack((points, np.ones((points.shape[0], 1))))
pts1_reproj = np.dot(P1, points.T).T
pts2_reproj = np.dot(P2, points.T).T

pts1_reproj /= pts1_reproj[:, 2][:, np.newaxis]
pts2_reproj /= pts2_reproj[:, 2][:, np.newaxis]

error1 = np.sqrt(np.mean(np.sum((pts1 - pts1_reproj[:, :2])**2, axis=1)))
error2 = np.sqrt(np.mean(np.sum((pts2 - pts2_reproj[:, :2])**2, axis=1)))

error = (error1 + error2) / 2.0
print("img1:")
print(error1)
print("img2:")
print(error2)
print("mean error")
print(error)