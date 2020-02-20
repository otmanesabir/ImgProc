from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def getParams(argss):
    op = 0
    # e = 1, d = 2
    if len(argss) < 5 or argv[1] not in ('e', 'd', 'o', 'c'):
        print("usage: e/d/o/c <SE file> <input file> <output file>")
        exit()

    if argv[1] == 'e':
        op = 1
    elif argv[1] == 'd':
        op = 2
    elif argv[1] == 'o':
        op = 3
    elif argv[1] == 'c':
        op = 4

    se = "../input/" + argv[2]
    infile = "../input/" + argv[3]
    outfile = "../output/" + argv[4]

    return op, se, infile, outfile


def genMatrix(file):
    return [[int(val) for val in line.split(',')] for line in file]


# EROSION USING MIN - ONLY TESTED FOR BINARY IMAGES
# The value of the output pixel is the minimum value of all the pixels in the input pixel's neighborhood.
# In a binary image, if any of the pixels is set to 0, the output pixel is set to 0.
# the min_val function finds smallest element in the sub-matrix

def erosion(matrix, se):
    res = np.array(matrix)
    mini_val = np.max(res)
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = min_val(matrix, i, j, se, mini_val)
    return ans


def min_val(matrix, a, b, se, mini_val):
    j = b
    for x in range(len(se)):
        for y in range(len(se[0])):
            if se[x][y] == 1:
                if 0 <= b < len(matrix[0]):
                    if matrix[a][b] < mini_val:
                        mini_val = matrix[a][b]
                    b += 1
                else:
                    break
        if 0 <= a < len(matrix) - 1:
            a += 1
            b = j
        else:
            break
    return mini_val


# DILATION USING MAX - BINARY IMAGES
# The value of the output pixel is the max value of the pixels that are in 
# neighborhood the size of the SE within the input image.
# max_val finds the largest element in the sub-matrix

def dilation(matrix, se):
    res = np.array(matrix)
    maxi = np.min(res)
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = max_val(matrix, i, j, se, maxi)
    return ans


def max_val(matrix, a, b, se, maxi_val):
    i = a
    j = b
    for x in range(len(se)):
        for y in range(len(se[0])):
            if se[x][y] == 1:
                if 0 <= b < len(matrix[0]):
                    if matrix[a][b] >= maxi_val:
                        maxi_val = matrix[a][b]
                    b += 1
                else:
                    break
        if 0 <= a < len(matrix) - 1:
            a += 1
            b = j
        else:
            break
    return maxi_val


# Uses reverse structuring element but same logic as previous dilation
# This should return the same result as a normal dilation because we only need one single check
# to cross the entire place

def max_val_reverse(matrix, a, b, se, maxi_val):
    j = b
    x = len(se) - 1
    while x >= 0:
        y = len(se[0]) - 1
        while y >= 0:
            if se[x][y] == 1:
                if 0 <= b < len(matrix[0]):
                    if matrix[a][b] > maxi_val:
                        maxi_val = matrix[a][b]
                    b += 1
                else:
                    break
            y -= 1
        if 0 <= a < len(matrix) - 1:
            a += 1
            b = j
            x -= 1
        else:
            break
    return maxi_val


def dilation_r(matrix, se):
    res = np.array(matrix)
    maxi = np.min(res)
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = max_val_reverse(matrix, i, j, se, maxi)
    return ans



# OPENING : EROSION + DILATION
# CLOSING : DILATION + EROSION
# I'm not sure about using a reversed structuring element?
# I found a website where nothing is mentioned about reversing the structuring element.
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm
# I still made a reverse dilation function which uses the SE from bottom-right
# It returns the same as a normal dilation (which kinda makes sense?)
# Imma try implementing a reverse erosion which should return a different/shifted output.

def opening(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    ans = erosion(matrix, se)
    return dilation(ans, se)


def closing(matrix, se):
    ans = dilation(matrix, se)
    return erosion(ans, se)


def main():
    operation, s, infile, outfile = getParams(argv)
    f = open(infile, "r")
    se = open(s, "r")
    matrix = genMatrix(f)
    se_matrix = genMatrix(se)
    plt.imsave('../output/imgs/erosion.png', np.array(erosion(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    plt.imsave('../output/imgs/dilation.png', np.array(dilation(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    plt.imsave('../output/imgs/opening.png', np.array(opening(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    plt.imsave('../output/imgs/closing.png', np.array(closing(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    if operation == 1:
        np.savetxt(outfile, erosion(matrix, se_matrix), fmt='%i', delimiter=',')
    elif operation == 2:
        np.savetxt(outfile, dilation_r(matrix, se_matrix), fmt='%i', delimiter=',')
    elif operation == 3: 
        np.savetxt(outfile, opening(matrix, se_matrix), fmt='%i', delimiter=',')
    else:
        np.savetxt(outfile, closing(matrix, se_matrix), fmt='%i', delimiter=',')
    
if __name__ == '__main__':
    main()
