import numpy as np
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

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
    plt.ylim(0, 1.5)
    plt.xlabel('Image [PIXELS]', fontsize=12)
    plt.ylabel('Time [S]', fontsize=12)
    plt.legend()
    plt.show()


def entropyCalc(X):
    uniq = set(X)
    P = [np.mean(X == x) for x in uniq]
    return sum(-p * np.log2(p) for p in P)

def miSingle(x, y, num_bins):
    hist = np.histogram2d(np.asarray(x).flatten(), np.asarray(y).flatten(), bins=num_bins)
    return entropyCalc(np.asarray(x).flatten()) + entropyCalc(np.asarray(y).flatten()) - entropyCalc(np.asarray(Image.fromarray(hist[0], 'RGB')).flatten())


def barChart():
    top = []
    i = 100
    image = np.round(np.random.rand(2000, 2000) * 255)
    image2 = np.round(np.random.rand(2000, 2000) * 255)
    while(i < 1300):
        start = timeit.default_timer()
        miSingle(image, image2, i)
        stop = timeit.default_timer()
        top.append((i, (stop - start)))
        i += 100
    labels, ys = zip(*top)
    xs = np.arange(len(labels))
    width = 0.5
    colors = ['#38B6FF' for _ in range (12)]
    plt.bar(xs, ys, width, align='center', color=colors)
    plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
    plt.xlabel('Bin Size', fontsize=12)
    plt.ylabel('Time [S]', fontsize=12)
    plt.savefig('neighbor-barchart.png')


def test_rnd(n):
    x = []
    y = []
    i = 100
    for _ in tqdm(range(n), desc="Mutual Information Calculator"):
        image1 = np.round(np.random.rand(i, i) * 255)
        image2 = np.round(np.random.rand(i, i) * 255)
        x.append(image1.size)
        start = timeit.default_timer()
        miSingle(image1, image2, 10)
        stop = timeit.default_timer()
        y.append(float(stop - start))
        i += 1
    makePlot(x, y)


if __name__ == "__main__":
    barChart()
    #test_rnd(2000)