import numpy as np

#
# Ugly Brute Force - Binary images only :
# Check if the structuring element's coord. sys. fits
# If it doesn't, then continue iterating
# If it does, then check the rest of the elements
# You can check the rest of the elements
# If all of the remaining elements fit, then mark the coordinate sys.
# If one element, doesn't fit, then raise flag and break.
# If flag is raised, then ignore everything and proceed to next iteration
# If not mark the coord. sys. with 1
# If we're at the border, then we simply skip the iteration because we assume the border is a 1
#

def erosion_bin(matrix, se):
    ans = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != se[0][0]: # skip iteration, since it doesn't match
                ans[i][j] = '0'
            elif matrix[i][j] == se[0][0]:
                x = i
                y = j
                flag = True
                for a in range(len(se)):
                    for b in range(len(se[0])):
                        if 0 <= y < len(matrix[0]) - 1: #check if we're on the border
                            y += 1
                        else:
                            y = j
                        if se[a][b] != matrix[x][y]:
                            flag = False
                            print("break")
                            break
                    if 0 <= x < len(matrix) - 1:
                        x += 1
                    else:
                        x = i
                    if not flag:
                        break
                if flag:
                    ans[i][j] = '1'
                else:
                    ans[i][j] = '0'
    print(ans)