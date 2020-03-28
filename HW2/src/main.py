import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from _collections import deque

MASK = -2
WSHD = 0
INIT = -1
INQUEUE = -3
LVLS = 256

def getNeighbors(height, width, pixel):
    i = max(0, pixel[0] - 1)
    j = min(height, pixel[0] + 2)

    x = max(0, pixel[1] - 1)
    y = min(width, pixel[1] + 2)

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

    return glob.reshape(2, -1).T

def getPixels(height, width):
    y_axis = np.zeros((height, width))
    for j in range(height):
        row = np.full((1, width), j)
        y_axis[j] = row

    x_axis = np.zeros((height, width))
    row = np.arange(0, height + 1) 
    for i in range(height):
        x_axis[i] = row
        
    plane = np.zeros((2, height, width))
    plane[0] = y_axis
    plane[1] = x_axis

    return plane.reshape(2, -1).T.astype(int)

         

def main():
    current_label = 0
    flag = False
    fifo = deque()
    image = np.array(Image.open("../input/ex.png"))
    height, width = image.shape
    total = height * width
    labels = np.full((height, width), INIT, np.int32)
    reshaped_image = image.reshape(total)

    # [y, x] pairs of pixel coordinates of the flattened image.
    pixels = getPixels(height, width)
    # Coordinates of neighbour pixels for each pixel.
    neighbours = np.array([getNeighbors(height, width, p) for p in pixels])
    print(pixels)


if __name__ == "__main__":
    main()
