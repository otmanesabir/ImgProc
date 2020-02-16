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


# Ugly Brute Force :
# Check if the structuring element's coord. sys. fits
# If it doesn't, then continue iterating
# If it does, then check the rest of the elements
# You can check the rest of the elements
# If all of the remaining elements fit, then mark the coordinate sys.
# If one element, doesn't fit, then raise flag and break.
# If flag is raised, then ignore everything and proceed to next iteration
# If not mark the coord. sys. with 1
# !!- Still need to implement border control. -!!
#

def erosion(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != se[0][0]:
                # skip iteration
                ans[i][j] = '0'
            elif matrix[i][j] == se[0][0]:
                x = i
                y = j
                flag = True
                for a in range(len(se)):
                    for b in range(len(se[0])):
                        if se[a][b] != matrix[x][y]:
                            flag = False
                            break
                        y += 1
                    if not flag:
                        break
                if flag:
                    ans[i][j] = '1'
                else:
                    ans[i][j] = '0'
    print(ans)


def main():
    operation, s, infile, outfile = getParams(argv)
    f = open(infile, "r")
    se = open(s, "r")
    matrix = genMatrix(f)
    se_matrix = genMatrix(se)
    print(matrix)
    print(se_matrix)


if __name__ == '__main__':
    main()
