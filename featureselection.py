import csv

def main():
    print('Welcome to Danial Beg\'s Feature Selection Algorithm!')
    f = input('\nType in the name of the file to test: ')
    f = open(f, 'r')

    # Reference for reading the file: https://docs.python.org/3/library/csv.html
    r = csv.reader(f, delimiter=' ', skipinitialspace=True)
    r1 = len(next(r))

    algo = int(input('Type the number of the algorithm you want to run:\n'
                     '\n1. Foward Selection'
                     '\n2. Backward Elimination\n\n'))
    print('\nThis dataset has ' + str(r1) + ' features.\n\n'
                                            'Please wait while I normalize the data!')

    search(r1)


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


if __name__ == "__main__":
    main()
