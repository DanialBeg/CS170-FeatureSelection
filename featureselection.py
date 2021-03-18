import csv
import math
import copy
import sys
import numpy as np
import time
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
    print('\nThis dataset has ' + str(r1-1) + ' features.\n\n')

    # Choosing a search technique
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
                # We need to deep copy as Python does pass by reference for function calls
                s_temp = copy.deepcopy(seen_features)

                # Temporarily add the row we're looking at into the set
                s_temp.add(j)

                # Reading in file into dataframe:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
                df = pd.read_fwf(inf, header=None)
                nr = len(df.index)

                # Deep copy the dataframe as it will get updated in the function
                df_c = df.copy(deep=True)[:-1]
                accur = leave_one_out_cross_validation(rl, s_temp, df_c)
                j_temp = j

                # Printing float as a nice percent:
                # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accur))

                # Update the max accuracy if we find a better accuracy, update the index too
                if accur >= max_accur:
                    max_accur = accur
                    f_accur = accur
                    finalj = j_temp

        # Add best column in the set, for real
        seen_features.add(finalj)
        s_c = copy.deepcopy(seen_features)
        d[f_accur] = s_c

        # Printing float as a nice percent:
        # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
        print('Feature set ' + str(seen_features) + ' was best, accuracy is ' + "{:.1%}".format(f_accur) + '\n')

    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) +
          ' which has an accuracy of ' + "{:.1%}".format(max(d.keys())) + '\n')
    print('Time used: ' + str(round(time.time()-start, 2)) + ' seconds.')


def backward_search(inf, rl):
    start = time.time()
    seen_features = set()
    for j in range(1, rl):
        seen_features.add(j)
    d = {}

    # Do the first iteration of a full set outside the loop
    s_temp = copy.deepcopy(seen_features)

    # Reading in file into dataframe:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
    df = pd.read_fwf(inf, header=None)
    nr = len(df.index)

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
                # We need to deep copy as Python does pass by reference for function calls
                s_temp = copy.deepcopy(seen_features)
                # Temporarily remove from the set
                s_temp.remove(j)

                # Reading in file into dataframe:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
                df = pd.read_fwf(inf, header=None)
                nr = len(df.index)

                # Deep copy the dataframe as it will get updated in the function
                df_c = df.copy(deep=True)

                accur = leave_one_out_cross_validation(rl, s_temp, df_c)
                j_temp = j

                # Printing float as a nice percent:
                # https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(accur))

                # Update the max accuracy if we find a better accuracy, update the index too
                if accur >= max_accur:
                    max_accur = accur
                    f_accur = accur
                    finalj =j_temp

        # Remove the column for real this time
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

    # Convert dataframe to Numpy array to make it run faster
    n = df.to_numpy()
    df_c = n

    # Make columns we're not looking at full of 0's
    for a in range(1, c):
        if a not in seen:
            df_c[:, a] = 0.0

    # Go through every row and compute the distances
    for k in range(nr):
        # Take a subsection of the array and its corresponding label
        obj_classify = df_c[k][1:c]
        label_obj_classify = df_c[k][0]

        nearest_n_dist = sys.maxsize
        nearest_n_loc = sys.maxsize

        for l in range(nr):
            dist = 0

            # If we're not on the same row, compute the distance and update it if it's closer
            if k != l:
                d = {}

                # Using Numpy to add the arrays in one line
                dist = math.sqrt(np.sum(np.power(obj_classify - df_c[l][1:c], 2)))

                # Made this <= instead of < to match the output from the briefing video
                if dist <= nearest_n_dist:
                    nearest_n_dist = dist
                    nearest_n_loc = l + 1
                    nearest_n_label = df_c[nearest_n_loc - 1][0]

        # If we correctly classify, increase the correct counter
        if label_obj_classify == nearest_n_label:
            corr_classified += 1
        # print('Object ' + str(k) + ' is class ' + str(int(float(label_obj_classify))))
        # print('Its nearest neighbor is ' + str(nearest_n_loc) + ' is class ' + str(int(float(nearest_n_label))))
    accur = corr_classified/nr
    return accur


if __name__ == "__main__":
    main()
