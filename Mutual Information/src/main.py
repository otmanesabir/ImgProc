import numpy as np
import math
import imageio
from PIL import Image
from tqdm import tqdm


def findHistogram(img):
    # filled soon


def getImg(path):
    img = np.array(Image.open(path))
    return img

def draft():
    #the draft we made the other day.
    mi = 0.0
    jointD = [[0.2, 0.1, 0.2],
        [0, 0.2, 0.1],
        [0.1, 0, 0.1]]
    mdX = [0.3, 0.3, 0.4]
    mdY = [0.5, 0.3, 0.2]
    for i in range(3):
        for j in range(3):
            print(i)
            if jointD[i][j] == 0:
                mi += 0
            else:
                a = mdX[j]
                b = mdY[i]
                mi += jointD[i][j]*math.log2(jointD[i][j]/(mdX[j]*mdY[i]))
    print(mi)

def main():
    findHistogram(getImg("../input/nuclei.png"))


if __name__ == '__main__':
    main()