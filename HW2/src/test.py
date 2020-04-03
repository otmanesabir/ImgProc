import numpy as np
import timeit
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from collections import deque
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar


MASK = -2
WSHD = 0
INIT = -1
INQUEUE = -3
LVLS = 256

def genMatrix(filename):
    with open(filename, 'r') as f:
        l = [[int(num) for num in line.split(',')] for line in f]
    imageio.imwrite('./temp.png', l)
    return l


def getNeighbors(height, width, pixel, n):
    i = max(0, pixel[0] - 1)
    j = min(height, pixel[0] + int(n / 4))

    x = max(0, pixel[1] - 1)
    y = min(width, pixel[1] + int(n / 4))

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

    return glob.reshape(2, -1).T.astype(int)


def getPixels(height, width):
    y_axis = np.zeros((height, width))
    for j in range(height):
        row = np.full((1, width), j)
        y_axis[j] = row

    x_axis = np.zeros((height, width))
    try:
        row = np.arange(0, height)
        for i in range(height):
            x_axis[i] = row
    except:
        try:
            row = np.arange(0, height + 1)
            for i in range(height):
                x_axis[i] = row
        except:
            row = np.arange(0, width)
            for i in range(height):
                x_axis[i] = row

    plane = np.zeros((2, height, width))
    plane[0] = y_axis
    plane[1] = x_axis

    return plane.reshape(2, -1).T.astype(int)


def watershed(img, n):
    pbar1 = ProgressBar()
    pbar2 = ProgressBar()
    current_label = 0
    flag = False
    que = deque()
    try:
        height, width = img.shape
        total = height * width
    except:
        try:
            img.transpose(2, 0, 1).reshape(3, -1)
            height, width = img.shape
            total = height * width
        except:
            print("Image format not supported")
            exit()

    labels = np.full((height, width), INIT, np.int32)  # Flat output image matrix, initialized with INIT
    flat_img = img.reshape(total)  # Flattening the image
    pixels = getPixels(height, width)  # Getting [y, x] pairs pairs of image
    print("Getting {} Neighbours...".format(n))
    neighbours = np.array(
        [getNeighbors(height, width, p, n) for p in pbar1(pixels)])  # Getting [y, x] pairs for neighbours of all pixels
    neighbours = neighbours.reshape(height, width)

    # Sorting pixels direct access
    idx = np.argsort(flat_img)
    sorted_img = flat_img[idx]
    sorted_pixels = pixels[idx]

    # Creating 256 evenly spaced grey value levels
    lvls = np.linspace(sorted_img[0], sorted_img[-1], LVLS)
    lvl_idx = []
    current_lvl = 0

    # Getting indices of pixels that have diff grey value levels
    for i in range(total):
        if sorted_img[i] > lvls[current_lvl]:
            while sorted_img[i] > lvls[current_lvl]:
                current_lvl += 1
            lvl_idx.append(i)
    lvl_idx.append(total)

    print("Creating Segmented Image...")
    start = 0
    for stop in pbar2(lvl_idx):

        # Masking all pixels at current level
        for pixel in sorted_pixels[start:stop]:
            labels[pixel[0], pixel[1]] = MASK
            # Adding the neighbours of existing basins to queue
            for nbr_pixel in neighbours[pixel[0], pixel[1]]:
                if labels[nbr_pixel[0], nbr_pixel[1]] >= WSHD:
                    labels[pixel[0], pixel[1]] = INQUEUE
                    que.append(pixel)
                    break

        # Extending basis
        while que:
            pixel = que.popleft()
            for nbr_pixel in neighbours[pixel[0], pixel[1]]:
                pixel_label = labels[pixel[0], pixel[1]]
                nbr_label = labels[nbr_pixel[0], nbr_pixel[1]]
                if nbr_label > 0:
                    if pixel_label == INQUEUE or (pixel_label == WSHD and flag):
                        labels[pixel[0], pixel[1]] = nbr_label
                    elif pixel_label > 0 and pixel_label != nbr_label:
                        labels[pixel[0], pixel[1]] = WSHD
                        flag = False
                elif nbr_label == WSHD:
                    if pixel_label == INQUEUE:
                        labels[pixel[0], pixel[1]] = WSHD
                        flag = True
                elif nbr_label == MASK:
                    labels[nbr_pixel[0], nbr_pixel[1]] = INQUEUE
                    que.append(nbr_pixel)

        # Looking for new minimas
        for pixel in sorted_pixels[start:stop]:
            if labels[pixel[0], pixel[1]] == MASK:
                current_label += 1
                que.append(pixel)
                labels[pixel[0], pixel[1]] = current_label
                while que:
                    q = que.popleft()
                    for r in neighbours[q[0], q[1]]:
                        if labels[r[0], r[1]] == MASK:
                            que.append(r)
                            labels[r[0], r[1]] = current_label

        start = stop

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

def test_single(i, n):
    image = np.round(np.random.rand(i, i) * 255)
    height, width = image.shape
    start = timeit.default_timer()
    watershed(image, n)
    stop = timeit.default_timer()
    x.append([height*width, stop - start])
    return

# KEEP THIS GLOBAL
x = [[]]

# MULTI THREADING TEST (FASTER)

def test_mtt(n):
    i = 1
    threads = []
    for _ in tqdm(range(300), desc="creating threads"):
        t = threading.Thread(target=test_single, args=(i, n))
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
def test_rnd(n):
    x = []
    y = []
    i = 1
    for _ in tqdm(range(300), desc="Watershed PT"):
        image = np.round(np.random.rand(i, i) * 255)
        x.append(image.size)
        start = timeit.default_timer()
        watershed(image, n)
        stop = timeit.default_timer()
        y.append(float(stop - start))
        i += 1
    makePlot(x, y)


def barChart():
    top = []
    i = 2
    image = np.round(np.random.rand(512, 512) * 255)
    while(i < 128):
        start = timeit.default_timer()
        watershed(image, i)
        stop = timeit.default_timer()
        top.append((i, (stop - start)))
        i *= 2
    labels, ys = zip(*top)
    xs = np.arange(len(labels))
    width = 0.5
    plt.bar(xs, ys, width, align='center', color=['#38B6FF', '#ED254E', '#9FFFCB', '#7B4B94', '#7D82B8', '#2E5266', '#545454'])
    plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(ys)
    plt.xlabel('Number of Neighbors', fontsize=12)
    plt.ylabel('Time [S]', fontsize=12)
    plt.savefig('neighbor-barchart.png')

if __name__ == "__main__":
    test_rnd(8)
    #test_mtt()
    #barChart()
    #test_img()



