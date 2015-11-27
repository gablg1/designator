from PIL import Image
import numpy as np
from scipy.special import cbrt

COLOR_INTENSITIES = 256
BAND_HISTOGRAM_DIM = COLOR_INTENSITIES * 3
BIN_SIZE = 10
K = COLOR_INTENSITIES / BIN_SIZE + 1

# Returns a binned color histogram of dimension K x K x K where
# K = 256/bin_size
def imgToBinnedHistogram(filepath):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    binned = img[:, :, :3] / BIN_SIZE
    M, N, D = binned.shape
    assert(D == 3)
    hist = np.zeros((K * K * K))
    for i in range(M):
        for j in range(N):
            r, g, b = tuple(binned[i, j])
            bucket = RGBToBin(r, g, b)
            hist[bucket] += 1

    # should use the below instead (sparse matrix)
    #return scipy.sparse.csr_matrix(hist.ravel())
    return hist

def RGBToBin(r, g, b):
    return r * K ** 2 + g * K + b

def binToRGB(bin_number):
    b = bin_number % K
    bin_number /= K
    g = bin_number % K
    r = bin_number / K
    return r, g, b

def binDistance(a, b):
    r1, g1, b1 = binToRGB(a)
    r2, g2, b2 = binToRGB(b)

    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

def binSquareDistance(a, b):
    r1, g1, b1 = binToRGB(a)
    r2, g2, b2 = binToRGB(b)

    return (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2

def imgToBandHistogram(filepath):
    return Image.open(filepath).histogram()[:BAND_HISTOGRAM_DIM]

def imgToArray(filepath):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    return img[:, :, :3].ravel()
