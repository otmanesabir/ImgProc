from sys import argv 

def getParams(argss):
    op = 0 #e = 1, d = 2

    if len(argss) < 5:
        print ("usage: e/d <input file> <output file>")
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

def main():
    operation, s, infile, outfile = getParams(argv)
    f = open(infile, "r")
    se = open(s, "r")
    matrix = genMatrix(f)
    seMatrix = genMatrix(se)
    print(matrix)
    print(seMatrix)


if __name__ == '__main__':
        main()

