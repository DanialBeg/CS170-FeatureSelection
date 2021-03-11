import csv
import math
import copy
import sys
import numpy as np
import pandas as pd


def main():
    print('Welcome to Danial Beg\'s Feature Selection Algorithm!')
    f = 'CS170_small_special_testdata__96.txt'
    print('\nType in the name of the file to test: ')
    in_f = f
    f = open(f, 'r')

    # Reference for reading the file: https://docs.python.org/3/library/csv.html
    r = csv.reader(f, delimiter=' ', skipinitialspace=True)
    r1 = len(next(r))

    algo = int(input('Type the number of the algorithm you want to run:\n'
                     '\n1. Foward Selection'
                     '\n2. Backward Elimination\n\n'))
    print('\nThis dataset has ' + str(r1) + ' features.\n\n')

    return search(in_f, r1)


def search(inf, rl):
    seen_features = set()
    d = {}

    for i in range(1, rl):
        max_accur = 0
        finalk = 0
        for k in range(1, rl):
            if k not in seen_features:
                s_temp = copy.deepcopy(seen_features)
                s_temp.add(k)
                accur = leave_one_out_cross_validation(inf, rl, s_temp)
                k_temp = k
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + str(accur))

                # print(seen_features)

            if accur >= max_accur:
                max_accur = accur
                f_accur = accur
                finalk = k_temp
                # print('K is ' + str(k))
        seen_features.add(finalk)
        s_c = copy.deepcopy(seen_features)
        d[f_accur] = s_c
        print('Feature set ' + str(seen_features) + ' was best, accuracy is ' + str(f_accur))
        # print(d)
    print('\n')
    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) +
          ' which has an accuracy of ' + str(max(d.keys())))
    # print(d)


def leave_one_out_cross_validation(inf, c, seen):
    # Reading in CSV into dataframe:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df = pd.read_csv(inf, header=None)
    nr = len(df.index)
    # print(c)
    corr_classified = 0

    for i in range(0, nr):
        df[0][i] = df[0][i].split()

    df_c = df.copy()

    for a in range(1, c):
        if a not in seen:
            for b in range(nr):
                df_c[0][b][a] = '0'
    # print(df_c[0][288])

    for k in range(nr):
        obj_classify = df_c[0][k][1:c]
        label_obj_classify = df_c[0][k][0]

        nearest_n_dist = sys.maxsize
        nearest_n_loc = sys.maxsize

        for l in range(nr):
            sum_mnhtn = 0
            if k != l:
                for m in range(len(obj_classify)):
                    sum_mnhtn += np.power(float(obj_classify[m]) - float(df_c[0][l][1:c][m]), 2)
                dist = math.sqrt(sum_mnhtn)
                # Made this <= instead of < to match the output from the briefing video
                if dist <= nearest_n_dist:
                    nearest_n_dist = dist
                    nearest_n_loc = l+1
                    nearest_n_label = df_c[0][nearest_n_loc-1][0]
        if label_obj_classify == nearest_n_label:
            corr_classified += 1
        # print('Object ' + str(k) + ' is class ' + str(int(float(label_obj_classify))))
        # print('Its nearest neighbor is ' + str(nearest_n_loc) + ' is class ' + str(int(float(nearest_n_label))))
    accur = corr_classified/nr
    return accur


if __name__ == "__main__":
    main()
