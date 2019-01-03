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
        print("Symbol: ", symbol)
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

def load_train_dev_test(folder_path):
    trainX, trainY = np.array([]).reshape((0,45,45)), np.array([])
    devX, devY = np.array([]).reshape((0,45,45)), np.array([])
    testX, testY = np.array([]).reshape((0,45,45)), np.array([])
    for filename in os.listdir(folder_path):
        data = np.load(pjoin(folder_path, filename))
        X, y = data['arr_0'], data['arr_1']
        X, y = shuffle_dataset(X, y)

        #import ipdb; ipdb.set_trace()
        m = X.shape[0]
        trainX = np.concatenate((trainX, X[:int(TRAIN * m)]))
        trainY = np.concatenate((trainY, y[:int(TRAIN * m)]))

        devX = np.concatenate((devX, X[int(TRAIN * m): int((TRAIN + DEV) * m)]))
        devY = np.concatenate((devY, y[int(TRAIN * m): int((TRAIN + DEV) * m)]))

        testX = np.concatenate((testX, X[int((TRAIN + DEV) * m):]))
        testY = np.concatenate((testY, y[int((TRAIN + DEV) * m):]))

    trainX, trainY = shuffle_dataset(trainX, trainY)
    devX, devY = shuffle_dataset(devX, devY)
    testX, testY = shuffle_dataset(testX, testY)
    return {
        "trainX": trainX,
        "trainY": trainY.reshape(-1,1),
        "devX": devX,
        "devY": devY.reshape(-1,1),
        "testX": testX,
        "testY": testY.reshape(-1,1)
    }


def load_dummy_set(examples, example_shape, categories):
    X = np.random.rand(examples, *example_shape)
    y = np.empty(examples, dtype='|S6')
    y[:] = [str(i % categories) for i in range(examples)]
    return X, y.reshape(-1,1)
