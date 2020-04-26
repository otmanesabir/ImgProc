from PIL import Image
import numpy as np


def entropyCalc(X):
    uniq = set(X)
    P = [np.mean(X == x) for x in uniq]
    return sum(-p * np.log2(p) for p in P)

def miSingle(x, y, num_bins):
    hist = np.histogram2d(np.asarray(x).flatten(), np.asarray(y).flatten(), bins=num_bins)
    return entropyCalc(np.asarray(x).flatten()) + entropyCalc(np.asarray(y).flatten()) - entropyCalc(np.asarray(Image.fromarray(hist[0], 'RGB')).flatten())

def main():
    flower = Image.open("../input/flower.png")
    puffin = Image.open("../input/puffin.jpg")
    miSingle(flower, puffin, 256)

if __name__ == '__main__':
    main()