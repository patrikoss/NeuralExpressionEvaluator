import os
from os.path import join as pjoin
from PIL import Image
import numpy as np

SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SYMBOLS += ["=", "-", "+", "times"]
#SYMBOLS += ["noise"]

SEED = 1
TRAIN = 0.8
DEV = 0.1

CATEGORIES = 14
SYMBOL_INPUT_SHAPE = (45,45,1)

def shuffle_dataset(X, y):
    rand_perm = np.random.permutation(X.shape[0])
    return X[rand_perm], y[rand_perm]

def load_symbols_dataset(symbols_dataset_npz_folder):
    X, y = np.array([]).reshape((0, 45, 45)), np.array([])
    symbol_npz_filenames = list(os.listdir(symbols_dataset_npz_folder))
    for symbol_npz_filename in symbol_npz_filenames:
        symbol_npz_filepath = pjoin(symbols_dataset_npz_folder, symbol_npz_filename)
        symbol = symbol_npz_filename[:-4]  # cut off .npz extension

        symbol_dataset = np.load(symbol_npz_filepath)
        symbol_X, symbol_y = symbol_dataset['X'], symbol_dataset['y']
        X = np.concatenate((X, symbol_X))
        y = np.concatenate((y, symbol_y))

    X, y = shuffle_dataset(X, y)
    return {
        "X":X,
        "y":y,
        "categories": len(symbol_npz_filenames)
    }

def show_image(np_array):
    Image.fromarray(np_array).show()
    #%matplotlib_inline
    #imshow(X[num].reshape(X[num].shape[0], X[num].shape[1]))
    #print(y[num])
    pass

