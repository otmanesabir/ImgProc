from sys import argv
import numpy as np
from _collections import deque
import cv2
from PIL import Image
import imageio
from scipy import ndimage
from progressbar import ProgressBar

MASK = -2
WSHD = 0
INIT = -1
INQUEUE = -3
LVLS = 256

def getParamsExperiments(argss):
    op = False
    if len(argss) < 4 or int(argv[2]) not in (1, 0):
        print("usage: <input file> <1 or 0 distanced> <output file.png/jpg>")
        exit()
    if argv[1].endswith(".txt"):op = True
    distanced = argv[2]
    infile = "../input/" + argv[1]
    outfile4 = "../output/4_" + argv[3]
    outfile8 = "../output/8_" + argv[3]
    outfile12 = "../output/12_" + argv[3]
    return op, int(distanced), infile, outfile4, outfile8, outfile12

def getParams(argss):
    op = False
    if len(argss) < 4 or int(argv[1]) not in (4, 8):
        print("usage: <input file> <4 or 8 neighbours> <output file.png/jpg>")
        exit()
    if argv[2].endswith(".txt"):op = True
    neighbours = argv[1]
    infile = "../input/" + argv[2]
    outfile = "../output/" + argv[3]
    return op, int(neighbours), infile, outfile

def distanceTransform(filename):
    with open(filename, 'r') as f:
        l = [[int(num) for num in line.split(',')] for line in f]
    img = ndimage.distance_transform_edt(l)
    imageio.imwrite('./temp_distanced.png', img)
    return img

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
            img.transpose(2,0,1).reshape(3,-1)
            height, width = img.shape
            total = height * width
        except:
            print("Image format not supported")
            exit()

    labels = np.full((height, width), INIT, np.int32) # Flat output image matrix, initialized with INIT
    flat_img = img.reshape(total) # Flattening the image
    pixels = getPixels(height, width) # Getting [y, x] pairs pairs of image
    print("Getting {} Neighbours...".format(n))
    neighbours = np.array([getNeighbors(height, width, p, n) for p in pbar1(pixels)]) # Getting [y, x] pairs for neighbours of all pixels
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
                if labels[nbr_pixel[0],nbr_pixel[1]] >= WSHD:
                    labels[pixel[0], pixel[1]] = INQUEUE
                    que.append(pixel)
                    break

        # Extending basis
        while que:
            pixel = que.popleft()
            for nbr_pixel in neighbours[pixel[0], pixel[1]]:
                pixel_label = labels[pixel[0], pixel[1]]
                nbr_label = labels[nbr_pixel[0],nbr_pixel[1]]
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


def main():
    textfile, d, infile, outfile4, outfile8, outfile12 = getParamsExperiments(argv)
    if textfile:
        if d:
            print("Input: Textfile, Distance Transform Required")
            distanceTransform(infile)
            img = np.array(Image.open("./temp_distanced.png"))
            imageio.imwrite(outfile4, watershed(img, 4))
            imageio.imwrite(outfile8, watershed(img, 8))
            imageio.imwrite(outfile12, watershed(img, 12))
        else:
            print("Input: Textfile, No Distance Transform")
            genMatrix(infile)
            img = np.array(Image.open("./temp.png"))
            imageio.imwrite(outfile4, watershed(img, 4))
            imageio.imwrite(outfile8, watershed(img, 8))
            imageio.imwrite(outfile12, watershed(img, 12))
    else:
        img = np.array(Image.open(infile))
        if (len(img.shape) > 2):
            print("Input: RGB Image, Converting to Grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("Input: GS Image")
        imageio.imwrite(outfile4, watershed(img, 4))
        imageio.imwrite(outfile8, watershed(img, 8))
        imageio.imwrite(outfile12, watershed(img, 12))

def experiments():
    textfile, d, infile, outfile4, outfile8 = getParamsExperiments(argv)
    img = np.array(Image.open(infile))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(grayed,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # # noise removal
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # # sure background area
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)

    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)

    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1

    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0

    # # imageio.imwrite("../output/openedTdilated_coins.jpg", markers)
    # markers = watershed(markers, 8)
    # img[markers == -1] = [255,0,0]
    # shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # sure_fg = np.uint8(sure_fg)
    # ret, markers = cv2.connectedComponents(sure_fg)
    # img = watershed(img, 4)
    kernel5 = np.ones((7,7),np.float32)/25
    # dst = cv2.filter2D(img,-1,kernel5)
    # blur = cv2.GaussianBlur(img,(7,7),0)
    blur = cv2.medianBlur(img,5)
    # blur = cv2.bilateralFilter(img,9,75,75)
    # (thresh, im_bw) = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # kernel = np.ones((2,2),np.uint8)
    # kernel2 = np.ones((3,3),np.uint8)
    # eroded = cv2.erode(thresh,kernel,iterations=1)
    # dilated = cv2.dilate(eroded,kernel2,iterations=2)
    # eroded = cv2.erode(dilated,kernel,iterations=1)
    # opening = cv2.morphologyEx(thresh,cv2S.MORPH_OPEN,kernel, iterations = 1)
    # closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    imageio.imwrite("./experiments/zebra/zebraMed.jpg", blur)
    imageio.imwrite("./experiments/zebra/zebraM_4.jpg", watershed(blur, 4))
    imageio.imwrite("./experiments/zebra/zebraM_8.jpg", watershed(blur, 8))
    imageio.imwrite("./experiments/zebra/zebraM_12.jpg", watershed(blur, 12))
    imageio.imwrite("./experiments/zebra/zebraM_16.jpg", watershed(blur, 16))
    imageio.imwrite("./experiments/zebra/zebraM_20.jpg", watershed(blur, 20))

if __name__ == "__main__":
    # experiments()
    main()
