import os
import time

def test():
    directory = '../input/test/512/'
    se = open('../input/SE2.txt', 'r')
    se_matrix = genMatrix(se)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            infile = os.path.join(directory, filename)
            print("current photo:" + filename, end=' ')
            matrix = plt.imread(infile, -1)
            outfile = '../output/images/test/' + filename
            # to get execution time only
            start_time = time.time()
            # erosion(matrix, se_matrix)
            plt.imsave(outfile, np.array(erosion(matrix, se_matrix)).reshape(len(matrix), len(matrix[0])), cmap=cm.gray)
            print(" in " + "--- %s seconds ---" % (time.time() - start_time))
        else:
            continue
    se.close()
