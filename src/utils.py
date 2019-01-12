import os
from os.path import join as pjoin
from PIL import Image
import numpy as np
import cv2

SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SYMBOLS += ["=", "-", "+", "times"]
SYMBOLS += ["background"]

SEED = 1
TRAIN = 0.8
DEV = 0.1

CATEGORIES = 15
INPUT_HEIGHT = 45
INPUT_WIDTH = 45
INPUT_CHANNEL = 1


def shuffle_dataset(X, y):
    rand_perm = np.random.permutation(X.shape[0])
    return X[rand_perm], y[rand_perm]


def rescale_dataset(X, target_height, target_width):
    newX = []
    for i in range(X.shape[0]):
        newX.append(cv2.resize(X[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR))
    return np.array(newX)


def load_symbols_dataset(symbols_dataset_npz_folder, target_height=INPUT_HEIGHT, target_width=INPUT_WIDTH):
    X, y = np.array([]).reshape((0, INPUT_HEIGHT, INPUT_WIDTH)), np.array([])
    symbol_npz_filenames = list(os.listdir(symbols_dataset_npz_folder))
    for symbol_npz_filename in symbol_npz_filenames:
        symbol_npz_filepath = pjoin(symbols_dataset_npz_folder, symbol_npz_filename)

        symbol_dataset = np.load(symbol_npz_filepath)
        symbol_X, symbol_y = symbol_dataset['X'], symbol_dataset['y']
        X = np.concatenate((X, symbol_X))
        y = np.concatenate((y, symbol_y))

    X, y = shuffle_dataset(X, y)

    return {
        "X": X,
        "y": y,
    }


def show_image(np_array):
    Image.fromarray(np_array).show()

