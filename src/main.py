from sys import argv
import numpy as np


def getParams(argss):
    op = 0
    # e = 1, d = 2
    if len(argss) < 5:
        print("usage: e/d <SE file> <input file> <output file>")
        exit()

    if argv[1] == 'e':
        operation = 1
    elif argv[1] == 'd':
        operation = 2

    se = "../input/" + argv[2]
    infile = "../input/" + argv[3]
    outfile = argv[4]

    return op, se, infile, outfile


def genMatrix(file):
    return [[int(val) for val in line.split(',')] for line in file]


# EROSION USING MIN - ONLY TESTED FOR BINARY IMAGES
# The value of the output pixel is the minimum value of all the pixels in the input pixel's neighborhood.
# In a binary image, if any of the pixels is set to 0, the output pixel is set to 0.
# the min_val function finds smallest element in the sub-matrix

def erosion(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ans[i][j] = min_val(matrix, i, j, se)
    print(ans)


def min_val(matrix, a, b, se):
    i = a
    j = b
    mini_val = 1
    for x in range(len(se)):
        for y in range(len(se[0])):
            if 0 <= b < len(matrix[0]) - 1:
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


def main():
    operation, s, infile, outfile = getParams(argv)
    f = open(infile, "r")
    se = open(s, "r")
    matrix = genMatrix(f)
    se_matrix = genMatrix(se)
    # view in matrix format
    print("Original")
    for row in matrix:
        print(' '.join(map(str, row)))
    print("Erosion :")
    erosion(matrix, se_matrix)
    # print(se_matrix)


if __name__ == '__main__':
    main()
