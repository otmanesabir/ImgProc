import numpy as np
import timeit
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from collections import deque
from ipython_genutils.py3compat import xrange


class Watershed(object):
    MASK = -2
    WSHD = 0
    INIT = -1
    INQE = -3

    def __init__(self, levels=256):
        self.levels = levels

    # Neighbour (coordinates of) pixels, including the given pixel.
    def _get_neighbors(self, height, width, pixel):
        # MATRIX HEIGHT
        i = max(0, pixel[0] - 1)
        j = min(height, pixel[0] + 1)

        # MATRIX WIDTH
        x = max(0, pixel[1] - 1)
        y = min(width, pixel[1] + 1)
        # x = 2
        # y = 5

        matrix = np.zeros(shape=(j - i, y - x))
        for t in range(j - i):
            temp = np.full((1, (y - x)), t + i)
            matrix[t] = temp
        temp = np.arange(x, y)
        matrix2 = np.zeros(shape=(j - i, y - x))
        for t in range(y - x):
            if t == j - i: break
            matrix2[t] = temp
        glob = np.zeros((2, j - i, y - x))
        glob[0] = matrix
        glob[1] = matrix2
        return glob.astype('int64').reshape(2, -1).T

    def apply(self, image):
        current_label = 0
        flag = False
        fifo = deque()

        height, width = image.shape
        total = height * width
        labels = np.full((height, width), self.INIT, np.int32)

        reshaped_image = image.reshape(total)
        # [y, x] pairs of pixel coordinates of the flattened image.
        pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
        # Coordinates of neighbour pixels for each pixel.
        neighbours = np.array([self._get_neighbors(height, width, p) for p in pixels])
        if len(neighbours.shape) == 3:
            # Case where all pixels have the same number of neighbours.
            neighbours = neighbours.reshape(height, width, -1, 2)
        else:
            # Case where pixels may have a different number of pixels.
            neighbours = neighbours.reshape(height, width)

        indices = np.argsort(reshaped_image)
        sorted_image = reshaped_image[indices]
        sorted_pixels = pixels[indices]

        # self.levels evenly spaced steps from minimum to maximum.
        levels = np.linspace(sorted_image[0], sorted_image[-1], self.levels)
        level_indices = []
        current_level = 0

        # Get the indices that deleimit pixels with different values.
        for i in range(total):
            if sorted_image[i] > levels[current_level]:
                # Skip levels until the next highest one is reached.
                while sorted_image[i] > levels[current_level]: current_level += 1
                level_indices.append(i)
        level_indices.append(total)

        start_index = 0
        for stop_index in level_indices:
            # Mask all pixels at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                labels[p[0], p[1]] = self.MASK
                # Initialize queue with neighbours of existing basins at the current level.
                for q in neighbours[p[0], p[1]]:
                    # p == q is ignored here because labels[p] < WSHD
                    if labels[q[0], q[1]] >= self.WSHD:
                        labels[p[0], p[1]] = self.INQE
                        fifo.append(p)
                        break

            # Extend basins.
            while fifo:
                p = fifo.popleft()
                # Label p by inspecting neighbours.
                for q in neighbours[p[0], p[1]]:
                    # Don't set lab_p in the outer loop because it may change.
                    lab_p = labels[p[0], p[1]]
                    lab_q = labels[q[0], q[1]]
                    if lab_q > 0:
                        if lab_p == self.INQE or (lab_p == self.WSHD and flag):
                            labels[p[0], p[1]] = lab_q
                        elif lab_p > 0 and lab_p != lab_q:
                            labels[p[0], p[1]] = self.WSHD
                            flag = False
                    elif lab_q == self.WSHD:
                        if lab_p == self.INQE:
                            labels[p[0], p[1]] = self.WSHD
                            flag = True
                    elif lab_q == self.MASK:
                        labels[q[0], q[1]] = self.INQE
                        fifo.append(q)

            # Detect and process new minima at the current level.
            for p in sorted_pixels[start_index:stop_index]:
                # p is inside a new minimum. Create a new label.
                if labels[p[0], p[1]] == self.MASK:
                    current_label += 1
                    fifo.append(p)
                    labels[p[0], p[1]] = current_label
                    while fifo:
                        q = fifo.popleft()
                        for r in neighbours[q[0], q[1]]:
                            if labels[r[0], r[1]] == self.MASK:
                                fifo.append(r)
                                labels[r[0], r[1]] = current_label

            start_index = stop_index

        return labels


def makePlot(x, y):
    # Scatter Plot
    # Best fit line
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    plt.plot(x, y, 'r+', label="Scatter", markersize=5)
    plt.plot(x_new, y_new, label="Best Fit", linewidth=2)
    plt.xlim([x[0] - 1, x[-1] + 1])
    plt.ylim(0, 320)
    plt.xlabel('Image [PIXELS]', fontsize=12)
    plt.ylabel('Time [S]', fontsize=12)
    plt.legend()
    plt.show()

def test_single(i):
    w = Watershed()
    image = np.round(np.random.rand(i, i) * 255)
    height, width = image.shape
    start = timeit.default_timer()
    w.apply(image)
    stop = timeit.default_timer()
    x.append([height*width, stop - start])
    return

# KEEP THIS GLOBAL
x = [[]]

# MULTI THREADING TEST (FASTER)

def test_mtt():
    i = 1
    threads = []
    for _ in tqdm(range(1000), desc="creating threads"):
        t = threading.Thread(target=test_single, args=(i,))
        threads.append(t)
        i += 1

    for a in tqdm(threads, desc="Starting Threads"):
        a.start()

    for a in tqdm(threads, desc="Joining threads"):
        a.join()

    new_x = []
    new_y = []
    for temp in x:
        i = 0
        for coord in temp:
            if i == 0:
                new_x.append(coord)
                i += 1
            else:
                new_y.append(coord)
    makePlot(new_x, new_y)


# RANDOM SIMPLE TEST
def test_rnd():
    w = Watershed()
    x = []
    y = []
    i = 1
    for _ in tqdm(range(1000), desc="Watershed PT"):
        image = np.round(np.random.rand(i, i) * 255)
        x.append(image.size)
        start = timeit.default_timer()
        w.apply(image)
        stop = timeit.default_timer()
        y.append(float(stop - start))
        i += 1
    makePlot(x, y)


if __name__ == "__main__":
    #test_mtt()
    #test_img()
    test_rnd()


