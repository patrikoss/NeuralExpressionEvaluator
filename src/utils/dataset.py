import numpy as np
import cv2
import os
from os.path import join as pjoin
from PIL import Image

TRAIN_DEBUG = 0.1
TRAIN = 0.8
DEV = 0.1


def shuffle_dataset(X, y):
    """
    Shuffle dataset
    """
    rand_perm = np.random.permutation(X.shape[0])
    return X[rand_perm], y[rand_perm]


def rescale_dataset(X, target_height, target_width):
    """
    Resizes images of customary (height, width, channels) into
    (target_height, target_width, channels)
    """
    assert len(X.shape) == 4
    newX = []
    for i in range(X.shape[0]):
        newX.append(cv2.resize(X[i], (target_height, target_width), interpolation=cv2.INTER_LINEAR))
    return np.array(newX)

def load_dataset_by_symbol(symbols_dataset_npz_folder, shuffle=True):
    """
    Given a path to a folder containing .npz files of symbols,
    for each file return the dataset corresponding to it
    """
    symbol_npz_filenames = list(os.listdir(symbols_dataset_npz_folder))
    symbols = dict()
    for symbol_npz_filename in symbol_npz_filenames:
        symbol_name = symbol_npz_filename[:-4]
        symbol_npz_filepath = pjoin(symbols_dataset_npz_folder, symbol_npz_filename)

        symbol_dataset = np.load(symbol_npz_filepath)

        if shuffle:
            symbol_X, symbol_y = shuffle_dataset(symbol_dataset['X'], symbol_dataset['y'])
        else:
            symbol_X, symbol_y = symbol_dataset['X'], symbol_dataset['y']

        symbols[symbol_name] = (symbol_X, symbol_y)
    return symbols

def load_dataset(symbols_dataset_npz_folder, shuffle=True):
    """
    Given a path to a folder containing .npz files of symbols,
    for each file return the dataset corresponding to it
    """
    symbols = load_dataset_by_symbol(symbols_dataset_npz_folder, shuffle=False)
    symbols_ordered = sorted(symbols.keys())
    symbols_X = np.concatenate([symbols[s][0] for s in symbols_ordered])
    symbols_y = np.concatenate([symbols[s][1] for s in symbols_ordered])

    if shuffle:
        symbols_X, symbols_y = shuffle_dataset(symbols_X, symbols_y)

    return {
        "X": symbols_X,
        "y": symbols_y
    }

def save_dataset(X, y, save_path):
    """
    Save dataset in a specified save_path
    """
    np.savez_compressed(save_path, X=X, y=y)


def transform_images_to_npzfile(symbol_image_folder_path, npzfilepath, shuffle=True):
    """
    Loads images from a folder of a single symbol images, and transform each content of the folder
    (set of images) into a .npz file
    """
    symbol = symbol_image_folder_path.split("/")[-1]
    X, counter = [], 0
    for image_name in os.listdir(symbol_image_folder_path):
        image = Image.open(pjoin(symbol_image_folder_path, image_name))

        X.append(np.array(image))
        counter += 1
        if counter % 100 == 0:
            print("Loaded {} images for symbol {}".format(counter, symbol))

    y = np.empty(len(X), dtype='|S6')  # data type: string
    y[:] = symbol
    X = np.array(X)
    if shuffle:
        X, y = shuffle_dataset(X, y)
    save_dataset(X, y, npzfilepath)

def partition_dataset(X, y, shuffle=True):
    """
    Partitions dataset into train, dev, test (and train_debug) partitions
    """
    if shuffle:
        X, y = shuffle_dataset(X, y)
    m = X.shape[0]
    train_boundary = int(TRAIN*m)
    dev_boundary = int((TRAIN+DEV)*m)
    train_debug_boundary = int(TRAIN_DEBUG*m)

    train_X, train_y = X[:train_boundary], y[:train_boundary]
    dev_X, dev_y = X[train_boundary:dev_boundary], y[train_boundary:dev_boundary]
    test_X, test_y = X[dev_boundary:], y[dev_boundary:]
    train_debug_X, train_debug_y = X[:train_debug_boundary], y[:train_debug_boundary]

    return {
        "train": (train_X, train_y),
        "dev": (dev_X, dev_y),
        "test": (test_X, test_y),
        "train_debug": (train_debug_X, train_debug_y)
    }


