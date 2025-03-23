# Machine Learning Project Group 4
# Authors: Julian Paagman, Pascal Boon, Niek Biesterbos & Mark den Ouden


import os
from preprocess import *
from collections import defaultdict


def main():
    # Preprocess the data
    data_path = 'lid_spaeng'
    data = preprocess_data(data_path)
    print(data)


if __name__ == '__main__':
    main()