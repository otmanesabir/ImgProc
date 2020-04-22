import numpy as np
import math

def draft():
    #the draft we made the other day.    mi = 0.0
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
    print("test")


if __name__ == '__main__':
    main()