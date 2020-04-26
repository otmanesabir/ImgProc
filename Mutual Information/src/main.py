import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from PIL import Image

plt.style.use('seaborn')

def splitImage(img):
    split_img = Image.Image.split(img)
    return split_img[0], split_img[1], split_img[2]

def cropImage(img):
    r, g, b = splitImage(img)
    w, h = img.size
    top = r.crop((20, 0, w-20, h))
    return top, [g.crop((40-x, 0, w-x, h)) for x in range(40, -1, -1)]

def entropyCalc(X):
    uniq = set(X)
    P = [np.mean(X == x) for x in uniq]
    return sum(-p * np.log2(p) for p in P)


def mutualInfoCalc(img, num_bins):
    top, bottoms = cropImage(img)
    mis = []
    for i in range(len(bottoms)):
        hist = np.histogram2d(np.asarray(top).flatten(), np.asarray(bottoms[i]).flatten(), bins=num_bins)
        hist_img = Image.fromarray(hist[0], 'RGB')
        mis.append(entropyCalc(np.asarray(top).flatten()) + entropyCalc(np.asarray(bottoms[i]).flatten()) - entropyCalc(np.asarray(hist_img).flatten()))
    return mis

def miSingle(x, y, num_bins):
    hist = np.histogram2d(np.asarray(x).flatten(), np.asarray(y).flatten(), bins=num_bins)
    return entropyCalc(np.asarray(x).flatten()) + entropyCalc(np.asarray(y).flatten()) - entropyCalc(np.asarray(Image.fromarray(hist[0], 'RGB')).flatten())

def binSizeChange(img):
    top, bottoms = cropImage(img)
    mis = []
    for i in range(1, 256):
        hist = np.histogram2d(np.asarray(top).flatten(), np.asarray(bottoms[20]).flatten(), bins=int(256/i))
        hist_img = Image.fromarray(hist[0], 'RGB')
        mis.append(entropyCalc(np.asarray(top).flatten()) + entropyCalc(np.asarray(bottoms[20]).flatten()) - entropyCalc(np.asarray(hist_img).flatten()))
    return mis


def mutualInformationTests():
    flower = Image.open("../input/flower.png")
    fig = plt.figure()
    plt.ylabel('Mutual Information')
    plt.xlabel('Image Translations')
    sns.lineplot([x for x in range(41)], mutualInfoCalc(flower, 256))
    plt.savefig("../output/flower_translations")
    puffin = Image.open("../input/puffin.jpg")
    fig = plt.figure()
    plt.ylabel('Mutual Information')
    plt.xlabel('Image Translations')
    sns.lineplot([x for x in range(41)], mutualInfoCalc(puffin, 256))
    plt.savefig("../output/puffin_translations")


def main():
    mutualInformationTests()



if __name__ == '__main__':
    main()