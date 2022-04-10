'''
Author: Lixin Luo  1951748@tongji.edu.cn
Reference: 
    D.G. Lowe, Distinctive image features from scale-invariant keypoints, IJCV' 04
        http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    OpenSIFT C source codes
        http://robwhess.github.io/opensift/
'''

import cv2
import numpy as np
import time


def conv2d(img, kernel):
    '''
    2D convolution

    @param img an image
    @param kernel convolution kernel

    @return Returns the result matrix of convolution
    '''
    assert len(img.shape) == 2  # channel == 1
    h, w = img.shape
    k = kernel.shape[0]
    p = (k - 1) // 2  # padding
    tmp = np.zeros((h + p * 2, w + p * 2))
    res = np.zeros((h, w))
    tmp[p:-p, p:-p] = img

    for i in range(h):
        for j in range(w):
            res[i, j] = np.sum(tmp[i: i + k, j: j + k] * kernel)

    return res


def downsample(img):
    '''
    Downsamples an image to a quarter of its size (half in each dimension)
    using nearest-neighbor interpolation

    @param img an image

    @return Returns an image whose dimensions are half those of img
    '''
    # smaller_img = np.zeros((img.shape[0] // 2, img.shape[1] // 2))

    # for i in range(smaller_img.shape[0]):
    #     for j in range(smaller_img.shape[1]):
    #         smaller_img[i][j] = (
    #             np.sum(img[2 * i: 2 * i + 2, 2 * j: 2 * j + 2]) + 2) // 4

    smaller_img = cv2.resize(
        img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

    return smaller_img


def gauss_conv(img, sigma):
    '''
    Convolve an image with a gaussian kernel
    @param img an image
    @param sigma sigma of gaussian kernel

    @return Returns an image whose dimensions are half those of img
    '''
    kernel_sz = (int(sigma * 3 + 1)) * 2 + 1
    c = int(sigma * 3 + 1)   # coordination of center
    kernel = np.zeros((kernel_sz, kernel_sz))
    for i in range(c + 1):
        for j in range(c + 1):
            kernel[i][j] = kernel[kernel_sz - i - 1][j] = kernel[i][kernel_sz - j - 1] \
                = kernel[kernel_sz - i - 1][kernel_sz - j - 1] \
                = np.exp(-((i - c) * (i - c) + (j - c) * (j - c)) / (2 * sigma * sigma))

    kernel = kernel / np.sum(kernel)   # normalize
    return conv2d(img, kernel)


def build_gauss_pyr(base, octvs, intvls, sigma):
    '''
    Builds Gaussian scale space pyramid from an image

    @param base base image of the pyramid
    @param octvs number of octaves of scale space
    @param intvls number of intervals per octave
    @param sigma amount of Gaussian smoothing per octave

    @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3) list
    '''
    sig = np.zeros(intvls + 3)
    k = 2.0 ** (1 / intvls)
    sig[0], sig[1] = sigma, sigma * np.sqrt(k * k - 1)
    for i in range(2, intvls + 3):
        sig[i] = sig[i - 1] * k

    gauss_pyr = [[None for i in range(intvls + 3)] for o in range(octvs)]

    for o in range(octvs):
        for i in range(intvls + 3):
            if o == 0 and i == 0:
                gauss_pyr[o][i] = base.copy()
            elif i == 0:
                gauss_pyr[o][i] = downsample(gauss_pyr[o - 1][i])
            else:
                gauss_pyr[o][i] = gauss_conv(gauss_pyr[o][i - 1], sig[i])
                # gauss_pyr[o][i] = cv2.GaussianBlur(gauss_pyr[o][i-1], (0,0), sigmaX=sig[i], sigmaY=sig[i])

    return gauss_pyr


def build_dog_pyr(gauss_pyr, octvs, intvls):
    '''
    Builds a difference of Gaussians scale space pyramid by subtracting adjacent
    intervals of a Gaussian pyramid

    @param gauss_pyr Gaussian scale-space pyramid
    @param octvs number of octaves of scale space
    @param intvls number of intervals per octave

    @return Returns a difference of Gaussians scale space pyramid as an
        octvs x (intvls + 2) list
    '''
    dog_pyr = [[gauss_pyr[o][i + 1] - gauss_pyr[o][i]
                for i in range(intvls + 2)] for o in range(octvs)]

    return dog_pyr


def is_extremum(dog_pyr, octv, intvl, r, c):
    '''
    Determines whether a pixel is a scale-space extremum by comparing it to it's
    3x3x3 pixel neighborhood.

    @param dog_pyr DoG scale space pyramid
    @param octv pixel's scale space octave
    @param intvl pixel's within-octave interval
    @param r pixel's image row
    @param c pixel's image col

    @return Returns 1 if the specified pixel is an extremum (max or min) among
        it's 3x3x3 pixel neighborhood.
    '''
    val = dog_pyr[octv][intvl][r, c]

    # check for maximun
    if val > 0:
        for i in range(intvl - 1, intvl + 2):
            for j in range(r - 1, r + 2):
                for k in range(c - 1, c + 2):
                    if (val < dog_pyr[octv][i][j, k]):
                        return False
    # check for minimum
    else:
        for i in range(intvl - 1, intvl + 2):
            for j in range(r - 1, r + 2):
                for k in range(c - 1, c + 2):
                    if (val > dog_pyr[octv][i][j, k]):
                        return False

    return True


def deriv3d(dog_pyr, octv, intvl, r, c):
    '''
    Computes the partial derivatives in x, y, and scale of a pixel in the DoG
    scale space pyramid.

    @param dog_pyr DoG scale space pyramid
    @param octv pixel's octave in dog_pyr
    @param intvl pixel's interval in octv
    @param r pixel's image row
    @param c pixel's image col

    @return Returns the vector of partial derivatives for pixel I
        { dI/dx, dI/dy, dI/ds }^T
    '''
    dx = (dog_pyr[octv][intvl][r, c + 1] - dog_pyr[octv][intvl][r, c - 1]) / 2
    dy = (dog_pyr[octv][intvl][r + 1, c] - dog_pyr[octv][intvl][r - 1, c]) / 2
    ds = (dog_pyr[octv][intvl + 1][r, c] - dog_pyr[octv][intvl - 1][r, c]) / 2

    return (dx, dy, ds)


def hessain3d(dog_pyr, octv, intvl, r, c):
    '''
    Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

    @param dog_pyr DoG scale space pyramid
    @param octv pixel's octave in dog_pyr
    @param intvl pixel's interval in octv
    @param r pixel's image row
    @param c pixel's image col

    @return Returns the Hessian matrix for pixel I
    '''
    d = dog_pyr[octv][intvl][r, c]
    dxx = dog_pyr[octv][intvl][r, c + 1] + \
        dog_pyr[octv][intvl][r, c - 1] - 2 * d
    dyy = dog_pyr[octv][intvl][r + 1, c] + \
        dog_pyr[octv][intvl][r - 1, c] - 2 * d
    dss = dog_pyr[octv][intvl + 1][r, c] + \
        dog_pyr[octv][intvl - 1][r, c] - 2 * d
    dxy = (dog_pyr[octv][intvl][r + 1, c + 1] + dog_pyr[octv][intvl][r - 1, c - 1]
           - dog_pyr[octv][intvl][r - 1, c + 1] - dog_pyr[octv][intvl][r + 1, c - 1]) / 4
    dxs = (dog_pyr[octv][intvl + 1][r, c + 1] + dog_pyr[octv][intvl - 1][r, c - 1]
           - dog_pyr[octv][intvl - 1][r, c + 1] - dog_pyr[octv][intvl + 1][r, c - 1]) / 4
    dys = (dog_pyr[octv][intvl + 1][r + 1, c] + dog_pyr[octv][intvl - 1][r - 1, c]
           - dog_pyr[octv][intvl + 1][r - 1, c] - dog_pyr[octv][intvl - 1][r + 1, c]) / 4

    hessian = np.array([[dxx, dxy, dxs],
                        [dxy, dyy, dys],
                        [dxs, dys, dss]])

    return hessian


def is_too_edge_like(dog_img, r, c, curv_thr):
    '''
    Determines whether a feature is too edge like to be stable by computing the
    ratio of principal curvatures at that feature.  Based on Section 4.1 of
    Lowe's paper.

    @param dog_img image from the DoG pyramid in which feature was detected
    @param r feature row
    @param c feature col
    @param curv_thr high threshold on ratio of principal curvatures

    @return Returns False if the feature at (r,c) in dog_img is sufficiently
        corner-like or True otherwise.
    '''
    # Hessian matrix
    d = dog_img[r, c]
    dxx = dog_img[r, c + 1] + dog_img[r, c - 1] - 2 * d
    dyy = dog_img[r + 1, c] + dog_img[r - 1, c] - 2 * d
    dxy = (dog_img[r + 1, c + 1] + dog_img[r - 1, c - 1] -
           dog_img[r - 1, c + 1] - dog_img[r + 1, c - 1]) / 4

    # compute the trace and det of Hessian
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy

    # Hessian should be semi positve definite or negative definite since the
    # sample point is a local extremum. -> det should be positive
    if det <= 0:
        return True

    # check ratio of principal curvatures
    if tr * tr / det < (curv_thr + 1.0) * (curv_thr + 1.0) / curv_thr:
        return False
    else:
        return True


def interp_contrast(dog_pyr, octv, intvl, r, c, xi, xr, xc):
    '''
    Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
    paper.

    @param dog_pyr difference of Gaussians scale space pyramid
    @param octv octave of scale space
    @param intvl within-octave interval
    @param r pixel row
    @param c pixel column
    @param xi interpolated subpixel increment to interval
    @param xr interpolated subpixel increment to row
    @param xc interpolated subpixel increment to col

    @param Returns interpolated contrast.
    '''
    dD = deriv3d(dog_pyr, octv, intvl, r, c)
    return dog_pyr[octv][intvl][r, c] + 0.5 * np.sum(dD * np.array([xc, xr, xi]))


def interp_extremum(dog_pyr, octv, intvl, r, c, intvls, contr_thr):
    '''
    Interpolates a scale-space extremum's location and scale to subpixel
    accuracy to form an image feature.  Rejects features with low contrast.
    Based on Section 4 of Lowe's paper.

    @param dog_pyr DoG scale space pyramid
    @param octv feature's octave of scale space
    @param intvl feature's within-octave interval
    @param r feature's image row
    @param c feature's image column
    @param intvls total intervals per octave
    @param contr_thr threshold on feature contrast

    @return Returns the feature resulting from interpolation of the given
        parameters or None if the given location could not be interpolated or
        if contrast at the interpolated loation was too low.
    '''

    i = 0
    # SIFT_MAX_INTERP_STEPS = 5
    # limit max interpolation step to ensure convergence
    while i < 5:
        # solve equation (3) in Lowe's paper
        grad = deriv3d(dog_pyr, octv, intvl, r, c)         # gradient of D
        hessian = hessain3d(dog_pyr, octv, intvl, r, c)    # hessian of D
        if np.linalg.det(hessian) == 0:
            return None
        xc, xr, xi = -np.matmul(np.linalg.inv(hessian), grad)

        if np.abs(xc) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        c += int(np.round(xc))
        r += int(np.round(xr))
        intvl += int(np.round(xi))

        # SIFT_IMG_BORDER = 5
        h, w = dog_pyr[octv][0].shape
        if intvl < 1 or intvl > intvls or c < 5 or r < 5 or c >= w - 5 or r >= h - 5:
            return None
        i += 1

    if i >= 5:
        return None

    contrast = interp_contrast(dog_pyr, octv, intvl, r, c, xi, xr, xc)
    if np.abs(contrast) < contr_thr / intvls:
        return None

    feat = {'x': (c + xc) * (2 ** octv), 'y': (r + xr) * (2 ** octv), 'r': r, 'c': c,
            'octv': octv, 'intvl': intvl, 'subintvl': xi}

    return feat


def scale_space_extrema(dog_pyr, intvls, contr_thr, curv_thr):
    '''
    Detects features at extrema in DoG scale space.  Bad features are discarded
    based on contrast and ratio of principal curvatures.

    @param dog_pyr DoG scale space pyramid
    @param intvls intervals per octave
    @param contr_thr low threshold on feature contrast
    @param curv_thr high threshold on feature ratio of principal curvatures

    @return Returns an list of detected features whose scales, orientations,
        and descriptors are yet to be determined.
    '''
    prelim_contr_thr = 0.5 * contr_thr / intvls
    features = []

    for o in range(len(dog_pyr)):
        h, w = dog_pyr[o][0].shape
        feature_map = np.zeros((intvls, h, w))
        # save the sample points that have been pushed into the features list

        for i in range(1, intvls + 1):
            for r in range(5, h - 5):      # SIFT_IMG_BORDER = 5
                for c in range(5, w - 5):
                    if np.abs(dog_pyr[o][i][r, c]) > prelim_contr_thr and is_extremum(dog_pyr, o, i, r, c):
                        feat = interp_extremum(
                            dog_pyr, o, i, r, c, intvls, contr_thr)
                        if feat != None:
                            if not is_too_edge_like(dog_pyr[feat['octv']][feat['intvl']],
                                                    feat['r'], feat['c'], curv_thr):
                                if feature_map[feat['intvl'] - 1, feat['r'], feat['c']] == 0:
                                    features.append(feat)
                                    feature_map[feat['intvl'] - 1, feat['r'], feat['c']] = 1

    return features


def calc_feature_scales(features, sigma, intvls):
    '''
    Calculates characteristic scale for each feature in a list.

    @param features list of features
    @param sigma amount of Gaussian smoothing per octave of scale space
    @param intvls intervals per octave of scale space
    '''
    for i in range(len(features)):
        intvl = features[i]['intvl'] + features[i]['subintvl']
        features[i]['scl'] = sigma * \
            (2 ** (features[i]['octv'] + intvl / intvls))
        features[i]['scl_octv'] = sigma * (2 ** (intvl / intvls))


def adjust_for_img_dbl(features):
    '''
    Halves feature coordinates and scale in case the input image was doubled
    prior to scale space construction.

    @param features list of features
    '''
    for i in range(len(features)):
        features[i]['x'] /= 2
        features[i]['y'] /= 2
        features[i]['scl'] /= 2


def draw_features(img, features):
    '''
    Draw circles to represent feature points. The center of the circle indicate 
    the spatial position of the point and the radius of the circle indicate 
    the characteristic scale of the point.

    @param img an image
    @param features list of features of the input image
    '''
    for feature in features:
        cv2.circle(img, (int(feature['x']), int(feature['y'])), int(feature['scl'] * 1.5), (0, 0, 255), lineType=cv2.LINE_AA)


if __name__ == '__main__':

    INTVLS = 3   # number of intervals per octave
    SIGMA = 1.6
    SIFT_INIT_SIGMA = 0.5
    CONTR_THR = 0.04
    CURV_THR = 10

    img_dir = './test.png'

    start = time.time()
    init_img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    octvs = int(np.log(min(init_img.shape)) /
                np.log(2) - 2)   # number of octaves

    # Double the image size and pre-smooth the image. Based on Section 4.1 of
    # Lowe's paper.
    init_img = cv2.resize(
        init_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR).astype('float32')
    sig_diff = np.sqrt(SIGMA * SIGMA - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4)
    init_img = gauss_conv(init_img, sig_diff)

    gauss_pyr = build_gauss_pyr(init_img, octvs, INTVLS, SIGMA)
    dog_pyr = build_dog_pyr(gauss_pyr, octvs, INTVLS)
    features = scale_space_extrema(dog_pyr, INTVLS, CONTR_THR, CURV_THR)
    print('%d keypoints detected in %.2f secs.' % (len(features), time.time() - start))

    img = cv2.imread(img_dir)
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    calc_feature_scales(features, SIGMA, INTVLS)
    # adjust_for_img_dbl(features)
    draw_features(img, features)
    
    cv2.imwrite('./features.png', img)
    print('Image with keypoints has been saved as features.png')
