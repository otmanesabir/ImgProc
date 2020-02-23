from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
import time


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

    se = "../submission_file/input/" + argv[2]
    infile = "../submission_file/input/" + argv[3]
    outfile = "../submission_file/output/" + argv[4]
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


# OPENING : EROSION + DILATION
# CLOSING : DILATION + EROSION
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

def opening(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    ans = erosion(matrix, se)
    return dilation(ans, se)


def closing(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    ans = dilation(matrix, se)
    return erosion(ans, se)


def main():
    operation, s, infile, outfile = getParams(argv)
    m_file = open(infile, "r")
    matrix = genMatrix(m_file)
    se = open(s, "r")
    se_matrix = genMatrix(se)
    #plt.imsave(outfile_e, np.array(erosion(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    #plt.imsave(outfile_d, np.array(dilation(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    #plt.imsave(outfile_o, np.array(opening(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    #plt.imsave(outfile_c, np.array(closing(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
    if operation == 1:
        np.savetxt(outfile, erosion(matrix, se_matrix), fmt='%i', delimiter=',')
    elif operation == 2:
        np.savetxt(outfile, dilation(matrix, se_matrix), fmt='%i', delimiter=',')
    elif operation == 3:
        np.savetxt(outfile, opening(matrix, se_matrix), fmt='%i', delimiter=',')
    elif operation == 4:
        np.savetxt(outfile, closing(matrix, se_matrix), fmt='%i', delimiter=',')


def test():
    directory = '../input/512/'
    se = open('../input/SE/SE10.txt', 'r')
    f = open("../time_tests/results_SE10.txt", "w+")
    my_time = 0.0
    cv2_time = 0.0
    count = 0
    se_matrix = genMatrix(se)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            infile = os.path.join(directory, filename)
            #print("current photo: " + filename, end=' ')
            matrix = plt.imread(infile, -1)
            outfile = '../output/images/test/' + filename
            # to get execution time only
            start_time = time.time()
            erosion(matrix, se_matrix)
            # plt.imsave(outfile, np.array(erosion(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
            f.write("%f" % (time.time() - start_time))
            f.write(",")
            my_time += (time.time() - start_time)
            #print(" in " + " %s seconds" % (time.time() - start_time), end=' ')
            kernel = np.ones((3, 3), np.uint8)
            start_time = time.time()
            cv2.erode(matrix, kernel)
            f.write("%f" % (time.time() - start_time))
            f.write("\n")
            cv2_time += (time.time() - start_time)
            #print("or in " + " %s seconds" % (time.time() - start_time))
        else:
            continue
        count += 1
    print("My average is : ", end=' ')
    print(my_time/count)
    print("CV2 average is : ", end=' ')
    print(cv2_time/count)
    se.close()


if __name__ == '__main__':
    test()
    #main()
