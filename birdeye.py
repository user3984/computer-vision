import numpy as np
import pickle
import cv2

with open('intrinsics.pkl', 'rb') as f:
    params = pickle.load(f)

K = params['mtx']
dist = params['dist']
newcammtx = params['newcammtx']

img = cv2.imread('./img.jpg')
undist_img = cv2.undistort(img, K, dist)
cv2.imwrite('undistorted.jpg', undist_img)

w = 11
h = 6
objp = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 100    # key points on the chessboard (WCS)
gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
u, v = img.shape[:2]

# find the corners
ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

# if all the corners are found
if ret == True:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Solve the equation using least squares
A = []
b = []
for pimg, pb in zip(np.flipud(corners[:, 0, :]), objp):
    u, v = pimg
    x, y = pb
    A.append(np.array([[u, v, 1, 0, 0, 0, -u*x, -v*x],
                    [0, 0, 0, u, v, 1, -u*y, -v*y]]))
    b.append([x, y])

A = np.concatenate(A, axis=0)
b = np.concatenate(b)
H1 = np.linalg.inv(np.matmul(A.T, A))
H2 = np.matmul(A.T, b)
h = np.matmul(H1, H2)
H = np.concatenate([h, (1,)]).reshape((3, 3))

translation = np.array([[1, 0,400],
                        [0, 1,400],
                        [0, 0, 1 ]])

birdseye = cv2.warpPerspective(undist_img, np.matmul(translation, H), (1800, 1400))

cv2.imwrite('birdseye.jpg', birdseye)