'''
Author: Lixin Luo  1951748@tongji.edu.cn
Reference: 
    D.G. Lowe, Distinctive image features from scale-invariant keypoints, IJCV' 04
        http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
'''

import cv2
import numpy as np
import random


def calc_cons_set(pairs, H, dist_thr):
    '''
    Compute the consensus set given a homography matrix and a set of pairs of points.

    @param pairs a list of pairs of points
    @param H a homography matrix
    @param dist_thr threshold on distance between estimate point and actual point

    @return Return the consensus set
    '''
    s = []   # consensus set
    for pair in pairs:
        u, v = pair[0]
        x, y = pair[1]
        x_, y_, z_ = np.matmul(H, (u, v, 1))  # estimate point based on H
        if z_ == 0:
            continue
        x_ /= z_
        y_ /= z_
        if (x - x_) * (x - x_) + (y - y_) * (y - y_) < dist_thr:
            s.append(pair)
    
    return s


def stitch_img(base, src, H):
    '''
    Stitch two image given the homography matrix.

    @param base the base image that does not transformed
    @param src the other source image
    @param H a nonsigular matrix representing perspective transformation from src to base

    @return Returns the panorama image after stitching
    '''
    y0, x0 = base.shape[:2]
    h, w = src.shape[:2]
    x1, y1, z1 = np.matmul(H, (0, 0, 1))
    x1 /= z1; y1 /= z1
    x2, y2, z2 = np.matmul(H, (w, 0, 1))
    x2 /= z2; y2 /= z2
    x3, y3, z3 = np.matmul(H, (w, h, 1))
    x3 /= z3; y3 /= z3
    x4, y4, z4 = np.matmul(H, (0, h, 1))
    x4 /= z4; y4 /= z4

    l = int(np.min((0, x0, x1, x2, x3, x4))) - 1
    r = int(np.max((0, x0, x1, x2, x3, x4))) + 1
    u = int(np.min((0, y0, y1, y2, y3, y4))) - 1
    d = int(np.max((0, y0, y1, y2, y3, y4))) + 1

    translation = np.array([[1, 0,-l],
                            [0, 1,-u],
                            [0, 0, 1]])

    pnrm = cv2.warpPerspective(src, np.matmul(translation, H), (r - l, d - u))

    pnrm[-u : -u + y0, -l : -l + x0] = base

    return pnrm


def calc_homography(pairs, ransac_max_steps, dist_thr, consensus_thr):
    '''
    Estimate homography matrix using RANSAC.

    @param pairs list of matched keypoint pairs (x, Hx)
    @param ransac_max_steps maximum iteration steps in RANSAC
    @param dist_thr threshold on distance between estimated points and actual points
    @param consensus_thr threshold on proportion of points in consensus set

    @return Returns the homography matrix H
    '''
    i = 0
    largest_size = 0        # largest size of consensus set
    A = None; b = None
    
    # limit max iteration steps to ensure convergence
    while i < ransac_max_steps:
        subset = random.sample(pairs, 4)
        _A = []
        _b = []
        for pair in subset:
            u, v = pair[0]
            x, y = pair[1]
            _A.append(np.array([[u, v, 1, 0, 0, 0, -u*x, -v*x],
                                [0, 0, 0, u, v, 1, -u*y, -v*y]]))
            _b.append([x, y])
        _A = np.concatenate(_A, axis=0)
        _b = np.concatenate(_b)
        # we assume H_{33} = 1
        if np.linalg.det(_A) == 0:
            continue

        # solve equation Ah = b
        _h = np.matmul(np.linalg.inv(_A), _b)
        _H = np.concatenate([_h, (1,)]).reshape((3, 3))

        cons_set = calc_cons_set(pairs, _H, dist_thr)
        if len(cons_set) >= largest_size:
            # re-estimate H using pairs in the consensus set
            _A = []
            _b = []
            for pair in cons_set:
                u, v = pair[0]
                x, y = pair[1]
                _A.append(np.array([[u, v, 1, 0, 0, 0, -u*x, -v*x],
                                [0, 0, 0, u, v, 1, -u*y, -v*y]]))
                _b.append([x, y])
            _A = np.concatenate(_A, axis=0)
            _b = np.concatenate(_b)
            if np.linalg.det(np.matmul(_A.T, _A)) != 0:
                largest_size = len(cons_set)
                A = _A; b = _b
                if len(cons_set) >= len(pairs) * consensus_thr:
                    break

        i += 1

    if i >= ransac_max_steps:
        print('RANSAC_MAX_STEPS has been reached!')

    print('Number of pairs in consensus set: ', largest_size)
    
    H1 = np.linalg.inv(np.matmul(A.T, A))
    H2 = np.matmul(A.T, b)
    h = np.matmul(H1, H2)
    H = np.concatenate([h, (1,)]).reshape((3, 3))

    return H


img1_dir = './img1.png'
img2_dir = './img2.png'
CONSENSUS_THR = 0.7
KEYPT_DIST_THR = 3
DESCR_DIST_THR = 150
RANSAC_MAX_STEPS = 1000

if __name__ == '__main__':

    img1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    sift = cv2.SIFT_create(edgeThreshold=10)

    keypoints1, descr1 = sift.detectAndCompute(img1, None)   # keypoints and descriptors
    keypoints2, descr2 = sift.detectAndCompute(img2, None)

    img = cv2.drawKeypoints(img1, keypoints1, img1,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('./cvsift.png', img)

    # match the keypoints in two images
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(descr1, descr2)

    img3 = cv2.drawMatches(img1, keypoints1, img2,
                           keypoints2, matches[:50], None, flags=2)
    cv2.imwrite('./cvmatch.png', img3)

    pairs = []

    for pair in matches:
        if pair.distance < DESCR_DIST_THR:
            pairs.append((keypoints1[pair.queryIdx].pt,
                         keypoints2[pair.trainIdx].pt))

    print('Number of pairs: ', len(pairs))

    H = calc_homography(pairs, RANSAC_MAX_STEPS, KEYPT_DIST_THR, CONSENSUS_THR)

    pnrm = stitch_img(img2, img1, H)

    cv2.imwrite('panorama.png', pnrm)
