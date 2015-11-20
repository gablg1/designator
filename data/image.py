from PIL import Image
import numpy as np

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
            hist[r, g, b] += 1

    # should use the below instead (sparse matrix)
    #return scipy.sparse.csr_matrix(hist.ravel())
    return hist.ravel()

def imgToBandHistogram(filepath):
    return Image.open(filepath).histogram()[:BAND_HISTOGRAM_DIM]

def imgToArray(filepath):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    return img[:, :, :3].ravel()
