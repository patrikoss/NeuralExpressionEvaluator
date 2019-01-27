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
    columns, rows = 3, len(images)
    fig = plt.figure()

    # ax enables access to manipulate each of subplots
    ax = []

    for row in range(rows):
        raw_img = images[row]

        encoded = encode_symbol(raw_img, intermediate_height, intermediate_width)
        decoded = decode_symbol(encoded, INPUT_HEIGHT, INPUT_WIDTH)

        ax.append(fig.add_subplot(rows, columns, columns*row+1))
        plt.imshow(raw_img, alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+2))
        plt.imshow(encoded, alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+3))
        plt.imshow(decoded, alpha=0.25, cmap='gray')

    plt.colorbar()
    plt.show()

# Example usage:
# dataset = np.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/0.npz")
# X = dataset['X']
# display_enc_dec_images( [X[0], X[1], X[2], X[3], X[4]], 20, 20 )

def generate_encoded_decoded_dataset(source_folder_path, target_folder_path,
                                     intermediate_shape):

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
                encoded = sl.encode_symbol(symbol, intermediate_height, intermediate_width)
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
                                     intermediate_shape=[(40, 20), (30, 20), (20, 20),
                                                         (70, 70), (70, 40), (70, 30), (70, 20)])


# Example usage:
#source_folder_path = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/"
#target_folder_path = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/encoded_decoded_npz_symbols"
#generate_encoded_decoded_dataset(source_folder_path, target_folder_path,
#                                 intermediate_shape=[(70,70), (20,20), (30,20), (40,20), (70,40)])
