import numpy as np


def calc_feature_oirs(features, gauss_pry):
    '''
    Computes a canonical orientation for each image feature in an array.  Based
    on Section 5 of Lowe's paper.  This function adds features to the array when
    there is more than one dominant orientation at a given feature location.

    @param features an array of image features
    @param gauss_pyr Gaussian scale space pyramid
    '''
    N = 36   # number of bins of the hist
    SIFT_ORI_PEAK_RATIO = 0.8

    sz = len(features)

    for i in range(sz):
        # computes a gradient orientation histogram at a specified pixel
        feature = features.pop(0)
        img = gauss_pry[feature['octv']][feature['intvl']]
        r, c = feature['r'], feature['c']
        sigma = 1.5 * feature['scl_octv']   # sigma of Gaussian window
        rad = int(3 * sigma + 0.5)          # radius of Guassian window
        hist = np.zeros(N)    # gradient orientation histogram with N bins

        r1, r2 = max(1, r - rad), min(img.shape[0] - 1, r + rad + 1)
        c1, c2 = max(1, c - rad), min(img.shape[1] - 1, c + rad + 1)

        # compute the gradient at each pixel
        dx = (img[r1: r2, c1 + 1: c2 + 1] - img[r1: r2, c1 - 1: c2 - 1]) / 2
        dy = (img[r1 + 1: r2 + 1, c1: c2] - img[r1 - 1: r2 - 1, c1: c2]) / 2
        mag = np.sqrt(dx * dx + dy * dy)
        ori = np.arctan2(dy, dx)
        bins = ((ori + np.pi) * N / (2 * np.pi) + 0.5).astype('int')
        bins[bins >= N] = 0

        # Gaussian window
        w = np.zeros((r2 - r1, c2 - c1))
        for i in range(0, r2 - r1):
            for j in range(0, c2 - c1):
                w[i, j] = (i - r + r1) ** 2 + (j - c + c1) ** 2
        w = np.exp(-w / (2 * sigma * sigma))

        mag = w * mag   # weighted magnitudes

        # add gradient magnitudes to corresponding bins
        for i in range(0, r2 - r1):
            for j in range(0, c2 - c1):
                hist[bin[i][j]] += mag[i][j]

        # smooth the orientation histogram
        for i in range(0, N):
            hist[i] = 0.25 * hist[(i - 1 + N) % N] + \
                0.5 * hist[i] + 0.25 * hist[(i + 1) % N]

        dominant_ori = np.max(hist)

        # adds features to the feature list for every local peak in a histogram greater than a
        # specified threshold.
        mag_thr = SIFT_ORI_PEAK_RATIO * dominant_ori
        for i in range(0, N):
            l = (i - 1 + N) % N
            r = (i + 1) % N
            if hist[i] > hist[l] and hist[i] > hist[r] and hist[i] >= mag_thr:
                # local peak that is greater than threshold

                # fit the hist with a quadratic function to interpolate to interpolate the
                # accurate local maxima
                b = i + 0.5 * (hist[l] - hist[r]) / \
                    (hist[l] + hist[r] - 2 * hist[i])
                if b < 0:
                    b = N + b
                elif b >= N:
                    b = b - N
                new_feat = feature.copy()
                new_feat['ori'] = (2 * np.pi * b / N) - np.pi
                features.append(new_feat)
