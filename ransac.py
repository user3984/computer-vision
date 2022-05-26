'''
Author: Lixin Luo  1951748@tongji.edu.cn
'''

import matplotlib.pyplot as plt
import numpy as np
import random


def calc_cons_set(points, k, b, dist_thr):
    '''
    Compute the consensus set given k, b and a set of points.

    @param points a list of points
    @param k the slope of the line
    @param b the intercept of the line
    @param dist_thr threshold on distance of a point to the line

    @return Return the consensus set
    '''
    s = []   # consensus set
    thr = dist_thr * np.sqrt(k * k + 1)
    for point in points:
        x, y = point
        if np.abs(k * x + b - y) < dist_thr:
            s.append(point)
    
    return s


def fit_line(points, ransac_max_steps, dist_thr, consensus_thr):
    '''
    Fit a line using RANSAC.

    @param points list of points
    @param ransac_max_steps maximum iteration steps in RANSAC
    @param dist_thr threshold on distance of a point to the line
    @param consensus_thr threshold on proportion of points in consensus set

    @return Returns k, b, consensus set
    '''
    i = 0
    largest_size = 0        # largest size of consensus set
    k = b = 0
    largest_cons_set = []
    
    # limit max iteration steps to ensure convergence
    while i < ransac_max_steps:
        subset = random.sample(points, 2)
        p1, p2 = subset
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            i += 1
            continue

        _k = (y1 - y2) / (x1 - x2)
        _b = y1 - _k * x1

        cons_set = calc_cons_set(points, _k, _b, dist_thr)
        if len(cons_set) >= largest_size:
            # re-estimate w, b using points in the consensus set
            X, Y = np.array(cons_set).T
            n = len(cons_set)
            _k = (n * (X * Y).sum() - X.sum() * Y.sum()) / (n * (X * X).sum() - X.sum() ** 2)
            _b = (Y.sum() - _k * X.sum()) / n
            largest_size = len(cons_set)
            k = _k; b = _b
            largest_cons_set = cons_set
            if len(cons_set) >= len(points) * consensus_thr:
                break

        i += 1

    if i >= ransac_max_steps:
        print('RANSAC_MAX_STEPS has been reached!')

    print('Number of points in consensus set: ', largest_size)

    return k, b, largest_cons_set


CONSENSUS_THR = 0.7
DIST_THR = 0.2
RANSAC_MAX_STEPS = 3000

if __name__ == '__main__':
    points = [(-2, 0), (0, 0.9), (2, 2.0), (3, 6.5), (4, 2.9), (5, 8.8), (6, 3.95), (8, 5.03), (10, 5.97), (12, 7.1), (13, 1.2), (14, 8.2), (16, 8.5), (18, 10.1)]
    k, b, cons_set = fit_line(points, RANSAC_MAX_STEPS, DIST_THR, CONSENSUS_THR)
    print('Equation of Line: y = %.2f x + %.2f' % (k, b))
    X, Y = np.array(points).T
    plt.scatter(X, Y, c='r')
    Xc, Yc = np.array([p for p in points if p in cons_set]).T
    plt.scatter(Xc, Yc, c='g')
    plt.plot([points[0][0], points[-1][0]], [k * points[0][0] + b, k * points[-1][0] + b])
    plt.show()
