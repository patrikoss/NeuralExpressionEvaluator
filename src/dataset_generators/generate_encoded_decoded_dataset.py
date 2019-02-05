import numpy as np
import os
from os.path import join as pjoin
from src.utils.dataset import save_dataset
from src.utils import symbol as sl
from src.utils.others import SEED, INPUT_HEIGHT, INPUT_WIDTH
from src.utils.symbol import encode_symbol, decode_symbol
import matplotlib.pyplot as plt


np.random.seed(SEED)


def display_enc_dec_images(images, intermediate_height, intermediate_width):
    """
    Displays a list of images after the process of encoding and decoding
    :param images: numpy array of images
    :param intermediate_height: int
    :param intermediate_width: int
    :return: None
    """
    columns, rows = 3, len(images)
    fig = plt.figure()

    # ax enables access to manipulate each of subplots
    ax = []

    for row in range(rows):
        raw_img = images[row]

        encoded = encode_symbol(raw_img, intermediate_height, intermediate_width, erode=True)
        decoded = decode_symbol(encoded, INPUT_HEIGHT, INPUT_WIDTH)

        ax.append(fig.add_subplot(rows, columns, columns*row+1))
        plt.imshow(raw_img, alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+2))
        plt.imshow(encoded, alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+3))
        plt.imshow(decoded, alpha=0.25, cmap='gray')

    plt.colorbar()
    plt.show()


def generate_encoded_decoded_dataset(source_folder_path, target_folder_path,
                                     intermediate_shape):
    """
    Transforms the symbols specified in the source folder into
    more appropriate shape and saves in the target folder
    :param source_folder_path: path to folder with input .npz symbols
    :param target_folder_path: path to folder with output folder
    :param intermediate_shape: intermediate shape (height, width) of the symbol
                                image
    :return: None
    """

    for filename in os.listdir(source_folder_path):

        symbol_name = filename[:-4]
        print("Resizing symbol: {}".format(symbol_name))

        source_dataset = np.load(pjoin(source_folder_path, filename))
        X, y = source_dataset['X'], source_dataset['y']
        count, newX, newY = 0, [], []

        for example in range(X.shape[0]):
            symbol = X[example]

            # append unmodified example
            newX.append(X[0]); newY.append(symbol_name)

            # append differently transformed images
            for intermediate_height, intermediate_width in intermediate_shape:
                encoded = sl.encode_symbol(symbol, intermediate_height, intermediate_width, erode=True)
                decoded = sl.decode_symbol(encoded, INPUT_HEIGHT, INPUT_WIDTH)
                newX.append(decoded); newY.append(symbol_name)

            count += 1
            if count % 100 == 0:
                print("Transformed {} images for symbol {}".format(count, symbol_name))

        newX = np.array(newX)
        newY = np.array(newY, dtype='|S6')  # data type: string
        save_dataset(newX, newY, pjoin(target_folder_path, filename))


def generate(source_folder_path, target_folder_path):
    generate_encoded_decoded_dataset(source_folder_path, target_folder_path,
                                     intermediate_shape=[(70,70), (20,20), (30,20), (40,20), (70,40)])


