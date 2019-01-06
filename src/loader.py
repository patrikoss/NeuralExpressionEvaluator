import os
from os.path import join as pjoin
import numpy as np
from PIL import Image
from .utils import SYMBOLS, SEED, TRAIN, DEV
from .utils import shuffle_dataset

np.random.seed(SEED)


def chunk_dataset(fraction_from, fraction_to, X, y):
    m = X.shape[0]
    chunk_X = X[int(fraction_from * m): int(fraction_to * m),]
    chunk_y = y[int(fraction_from * m): int(fraction_to * m),]
    return {
        "X": chunk_X,
        "y": chunk_y
    }


def tranform_symbol_images_to_npzfile(symbol, symbol_image_folder_path, npzfilepath):
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
    X, y = shuffle_dataset(X, y)
    save_dataset(X, y, npzfilepath)


def load_raw_image_symbols_dataset(folder_path):
    """Returns the dataset in a numpy array"""
    data = dict()
    for symbol in SYMBOLS:
        print("Loading symbol: ", symbol)
        data[symbol] = list()
        symbol_data_path = pjoin(folder_path, symbol)
        counter = 0
        for image_name in os.listdir(symbol_data_path):
            image = Image.open(pjoin(symbol_data_path, image_name))
            data[symbol].append(np.array(image))
            counter += 1
            if counter % 100 == 0:
                print("Loaded {} images".format(counter))

    cls = dict()
    for symbol in SYMBOLS:
        cls[symbol] = np.empty(len(data[symbol]), dtype='|S6')
        cls[symbol][:] = symbol
        data[symbol] = np.array(data[symbol])

        data[symbol], cls[symbol] = shuffle_dataset(data[symbol], cls[symbol])

    return data, cls


def save_dataset(X, y, save_path):
    np.savez_compressed(save_path, X=X, y=y)


def load_dataset(load_path):
    dataset = np.load(load_path)
    return {
        'X': dataset['X'],
        'y': dataset['y']
    }
