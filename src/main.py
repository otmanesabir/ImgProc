from sys import argv
import numpy as np


def getParams(argss):
    op = 0
    # e = 1, d = 2
    if len(argss) < 5 or argv[1] not in ('e', 'd'):
        print("usage: e/d <SE file> <input file> <output file>")
        exit()

    if argv[1] == 'e':
        op = 1
    elif argv[1] == 'd':
        op = 2

    se = "../input/" + argv[2]
    infile = "../input/" + argv[3]
    outfile = argv[4]

    return op, se, infile, outfile


def genMatrix(file):
    return [[int(val) for val in line.split(',')] for line in file]


def min_val(matrix, a, b, se):
    j = b
    mini_val = 255
    for x in range(len(se)):
        for y in range(len(se[0])):
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
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)): 
        for j in range(len(matrix[0])):
            ans[i][j] = max_val(matrix, i, j, se)
    print(ans)
            
def max_val(matrix, a, b, se): 
    i = a
    j = b
    max_val =  1
    for x in range(len(se)): 
        for y in range(len(se[0])):
            # print(max_val, i, j)
            if 0 <= b < len(matrix[0]):
                if matrix[i][j] > max_val: 
                    max_val = matrix[i][j]
                b += 1
            else:
                # max_val = 0
                break
        if 0 <= a < len(matrix) - 1: 
            a += 1
            b = j
        else:
            break   
    return max_val
                        

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


# reverse a structuring element

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

    if operation == 1:
        print("Erosion:")
        erosion(matrix, se_matrix)
    else:
        print("Dilation:")
        dilation(matrix, se_matrix)

if __name__ == '__main__':
    main()
