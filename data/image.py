from PIL import Image
import numpy as np
from scipy.special import cbrt

COLOR_INTENSITIES = 256
BAND_HISTOGRAM_DIM = COLOR_INTENSITIES * 3

# Returns a binned color histogram of dimension K x K x K where
# K = 256/bin_size
def imgToBinnedHistogram(filepath, bin_size=10):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    binned = img[:, :, :3] / bin_size
    M, N, D = binned.shape
    assert(D == 3)
    K = COLOR_INTENSITIES / bin_size + 1
    hist = np.zeros((K, K, K))
    for i in range(M):
        for j in range(N):
            r, g, b = tuple(binned[i, j])
            bucket = RGBToBin(r, g, b, K)
            hist[bucket] += 1

    # should use the below instead (sparse matrix)
    #return scipy.sparse.csr_matrix(hist.ravel())
    return hist

def RGBToBin(r, g, b, dim):
    K = cbrt(dim)
    return r * K ** 2 + g * K + b

def binToRGB(bin_number, dim):
    K = cbrt(dim)
    b = bin_number % K
    bin_number /= K
    g = bin_number % K
    r = bin_number / K
    return r, g, b

def imgToBandHistogram(filepath):
    return Image.open(filepath).histogram()[:BAND_HISTOGRAM_DIM]

def imgToArray(filepath):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    return img[:, :, :3].ravel()
