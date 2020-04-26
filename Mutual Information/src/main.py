import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


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

def splitImage(img):
    split_img = Image.Image.split(img)
    return split_img[0], split_img[1], split_img[2]

def cropImage(img):
    r, g, b = splitImage(img)
    w, h = img.size
    top = r.crop((20, 0, w-20, h))
    return top, [g.crop((40-x, 0, w-x, h)) for x in range(40, -1, -1)]

def binSizeChange(img):
    top, bottoms = cropImage(img)
    mis = []
    for i in range(1, 256):
        hist = np.histogram2d(np.asarray(top).flatten(), np.asarray(bottoms[20]).flatten(), bins=int(256/i))
        hist_img = Image.fromarray(hist[0], 'RGB')
        mis.append(entropyCalc(np.asarray(top).flatten()) + entropyCalc(np.asarray(bottoms[20]).flatten()) - entropyCalc(np.asarray(hist_img).flatten()))
    return mis

def mutualInformationTests(puffin):
    fig = plt.figure()
    plt.ylabel('Mutual Information')
    plt.xlabel('Image Translations')
    sns.lineplot([x for x in range(41)], mutualInfoCalc(puffin, 256))
    plt.savefig("../output/puffin_translations")
    fig = plt.figure()
    plt.ylabel('Mutual Information')
    plt.xlabel('Number of Bins')
    sns.lineplot([x for x in range(255)], binSizeChange(puffin))
    plt.savefig("../output/puffin_binSize")

def seperatorImg(image):
    titles = ['Flower', 'Red channel', 'Green channel', 'Blue channel']
    cmaps = [None, plt.cm.gray, plt.cm.gray, plt.cm.gray]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    objs = zip(axes, (image, *image.transpose(2, 0, 1)), titles, cmaps)
    for ax, channel, title, cmap in objs:
        ax.imshow(channel, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.savefig('../output/RGB1-Flower.png')

def main():
    puffin = Image.open("../input/puffin.jpg")
    mutualInformationTests(puffin)

if __name__ == '__main__':
    main()