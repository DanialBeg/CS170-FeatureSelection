import csv
import math
import sys
import numpy as np
import pandas as pd


def main():
    print('Welcome to Danial Beg\'s Feature Selection Algorithm!')
    f = input('\nType in the name of the file to test: ')
    in_f = f
    f = open(f, 'r')

    # Reference for reading the file: https://docs.python.org/3/library/csv.html
    r = csv.reader(f, delimiter=' ', skipinitialspace=True)
    r1 = len(next(r))

    algo = int(input('Type the number of the algorithm you want to run:\n'
                     '\n1. Foward Selection'
                     '\n2. Backward Elimination\n\n'))
    print('\nThis dataset has ' + str(r1) + ' features.\n\n')

    # search(r1)
    return kfold(in_f, r1)


def search(rl):

    seen_features = set()
    s_index = 0

    for i in range(1, rl):
        print('On level number ' + str(i))
        max_accur = 0

        for k in range(1, rl):
            if k not in seen_features:
                print('Considering adding the ' + str(k) + ' feature')
                accur = leave_one_out_cross_validation()
                accur = 2

                if accur > max_accur:
                    max_accur = accur
                    seen_features.add(k)
                    add_feature = k
        print('On level ' + str(i) + ' I added feature ' + str(add_feature)
              + ' to the current set.')


def leave_one_out_cross_validation():
    return 0


def kfold(inf, c):
    # Reading in CSV into dataframe:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df = pd.read_csv(inf, header=None)
    nr = len(df.index)

    for i in range(0, nr):
        df[0][i] = df[0][i].split()

    for k in range(1, nr+1):
        obj_classify = df[0][k-1][1:c]
        label_obj_classify = df[0][k-1][0]
        # print(label_obj_classify)

        nearest_n_dist = sys.maxsize
        nearest_n_loc = sys.maxsize

        for l in range(1, nr+1):
            if k != l:
                # print(str(k) + ' ' + str(l))
                sum_mnhtn = 0
                for m in range(len(obj_classify)):
                    sum_mnhtn += int(np.power(float(obj_classify[m]) - float(df[0][l-1][1:c][m]), 2))
                    # print(df[0][l-1][1:c][m])
                # print('Dist')
                dist = math.sqrt(sum_mnhtn)
                # print(str(l) + ' ' + str(dist))
                # Made this <= instead of < to match the output from the briefing video
                if dist <= nearest_n_dist:
                    nearest_n_dist = dist
                    nearest_n_loc = l
                    nearest_n_label = df[0][nearest_n_loc-1][0]
        print('Object ' + str(k) + ' is class ' + str(int(float(label_obj_classify))))
        print('Nearest neighbor is ' + str(nearest_n_loc) + ' is class ' + str(int(float(nearest_n_label))))

    return 0


if __name__ == "__main__":
    main()
