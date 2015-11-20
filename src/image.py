from PIL import Image
import numpy as np


class SiteImage():
    COLOR_INTENSITIES = 256
    BAND_HISTOGRAM_DIM = COLOR_INTENSITIES * 3

    def __init__(self, filename, amount, cut, big):
        path = data.getDataDir(amount=amount, cut=cut, big=big)
        self.image = Image.open('%s/%s' % (path, filenam))

    # Returns a binned color histogram of dimension K x K x K where
    # K = 256/bin_size
    def toHistogram(filepath, bin_size=10):
        try:
            img = np.array(self.image)
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
        return hist.ravel()

    def toBandHistogram(self):
        return self.image.histogram()[:BAND_HISTOGRAM_DIM]

    def toArray():
        try:
            img = np.array(self.image)
        except IOError as e:
            print e
        return img[:, :, :3].ravel()
