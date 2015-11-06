from PIL import Image
import numpy as np

# Returns a binned color histogram of dimension K x K x K where
# K = 256/bin_size
def imgToHistogram(filepath, bin_size=10):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    binned = img[:, :, :3] / bin_size
    M, N, D = binned.shape
    assert(D == 3)
    INTENSITIES = 256
    K = INTENSITIES / bin_size + 1
    hist = np.zeros((K, K, K))
    for i in range(M):
        for j in range(N):
            r, g, b = tuple(binned[i, j])
            hist[r, g, b] += 1
    return hist

def imgToArray(filepath):
    try:
        img = np.array(Image.open(filepath))
    except IOError as e:
        print e
    return img[:, :, :3].ravel()
