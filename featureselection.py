import csv
import math
import copy
import sys
import numpy as np
import time
import pandas as pd


def main():
    print('Welcome to Danial Beg\'s Feature Selection Algorithm!')
    f = 'CS170_small_special_testdata__95.txt'
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

    if algo == 1:
        return forward_search(in_f, r1)
    elif algo == 2:
        return backward_search(in_f, r1)
    return 'Incorrect input'


def forward_search(inf, rl):
    start = time.time()
    seen_features = set()
    d = {}

    for i in range(1, rl):
        max_accur = 0
        finalj = 0
        for j in range(1, rl):
            if j not in seen_features:
                s_temp = copy.deepcopy(seen_features)
                s_temp.add(j)

                # Reading in CSV into dataframe:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
                df = pd.read_csv(inf, header=None)
                nr = len(df.index)

                for k in range(0, nr):
                    df[0][k] = df[0][k].split()

                df_c = df.copy(deep=True)
                accur = leave_one_out_cross_validation(rl, s_temp, df_c)
                j_temp = j

                # Printing float as a nice percent:
                # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accur))

            if accur >= max_accur:
                max_accur = accur
                f_accur = accur
                finalj = j_temp
        seen_features.add(finalj)
        s_c = copy.deepcopy(seen_features)
        d[f_accur] = s_c

        # Printing float as a nice percent:
        # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
        print('Feature set ' + str(seen_features) + ' was best, accuracy is ' + "{:.1%}".format(f_accur) + '\n')
    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) +
          ' which has an accuracy of ' + "{:.1%}".format(max(d.keys())) + '\n')
    print('Time used: ' + str(round(time.time()-start, 2)) + ' seconds.')
    # print(d)


def backward_search(inf, rl):
    start = time.time()
    seen_features = set()
    for j in range(1, rl):
        seen_features.add(j)
    d = {}

    s_temp = copy.deepcopy(seen_features)

    # Reading in CSV into dataframe:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df = pd.read_csv(inf, header=None)
    nr = len(df.index)

    for k in range(0, nr):
        df[0][k] = df[0][k].split()

    df_c = df.copy(deep=True)

    accur = leave_one_out_cross_validation(rl, s_temp, df_c)

    # Printing float as a nice percent:
    # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
    print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accur))
    print('Feature set ' + str(s_temp) + ' was best, accuracy is ' + "{:.1%}".format(accur) + '\n')

    for i in range(2, rl):
        max_accur = 0
        finalj = 0

        for j in range(1, rl):
            if j in seen_features:
                s_temp = copy.deepcopy(seen_features)
                s_temp.remove(j)

                # Reading in CSV into dataframe:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
                df = pd.read_csv(inf, header=None)
                nr = len(df.index)

                for k in range(0, nr):
                    df[0][k] = df[0][k].split()

                df_c = df.copy(deep=True)

                accur = leave_one_out_cross_validation(rl, s_temp, df_c)
                j_temp = j

                # Printing float as a nice percent:
                # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accur))

            if accur >= max_accur:
                max_accur = accur
                f_accur = accur
                finalj =j_temp
        seen_features.remove(finalj)
        s_c = copy.deepcopy(seen_features)
        d[f_accur] = s_c

        # Printing float as a nice percent:
        # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
        print('Feature set ' + str(seen_features) + ' was best, accuracy is ' + "{:.1%}".format(f_accur) + '\n')
    print('\n')
    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) +
          ' which has an accuracy of ' + "{:.1%}".format(max(d.keys())) + '\n')
    print('Time used: ' + str(round(time.time()-start, 2)) + ' seconds.')


def leave_one_out_cross_validation(c, seen, df_c):
    nr = len(df_c.index)
    corr_classified = 0

    df = df_c.copy(deep=True)

    n = df.to_numpy()
    df_c = n

    for a in range(1, c):
        if a not in seen:
            for b in range(nr):
                df_c[b][0][a] = '0'

    # print(df_c[0][0][1:c])

    for k in range(nr):
        obj_classify = df_c[k][0][1:c]
        label_obj_classify = df_c[k][0][0]

        nearest_n_dist = sys.maxsize
        nearest_n_loc = sys.maxsize

        for l in range(nr):
            sum_mnhtn = 0
            if k != l:
                for m in range(len(obj_classify)):
                    # if float(df_c[l][0][m+1]) != 0.0:
                    #     print(float(obj_classify[m]))
                    # float(df_c[l][0][1:c][m]
                    sum_mnhtn += (float(obj_classify[m]) - float(df_c[l][0][m+1]))**2
                dist = math.sqrt(sum_mnhtn)
                # Made this <= instead of < to match the output from the briefing video
                if dist <= nearest_n_dist:
                    nearest_n_dist = dist
                    nearest_n_loc = l+1
                    nearest_n_label = df_c[nearest_n_loc-1][0][0]
        if label_obj_classify == nearest_n_label:
            corr_classified += 1
        # print('Object ' + str(k) + ' is class ' + str(int(float(label_obj_classify))))
        # print('Its nearest neighbor is ' + str(nearest_n_loc) + ' is class ' + str(int(float(nearest_n_label))))
    accur = corr_classified/nr
    return accur


if __name__ == "__main__":
    main()
