import cv2
import numpy as np
import os
from src.utils import show_image
from PIL import Image
import matplotlib.pyplot as plt

INTERPOLATION_ENCODE = cv2.INTER_LINEAR
INTERPOLATION_DECODE = cv2.INTER_LINEAR


def decode(raw_box, rescaled_height=45, rescaled_width=45):
    _, raw_box = cv2.threshold(raw_box, 127, 255, cv2.THRESH_BINARY)
    raw_box = cv2.resize(raw_box, (rescaled_width, rescaled_height), interpolation=cv2.INTER_LINEAR)
    _, raw_box = cv2.threshold(raw_box, 127, 255, cv2.THRESH_BINARY)
    raw_box = cv2.dilate(raw_box, kernel=np.ones((3,3), dtype=np.uint8), iterations=1)

    return raw_box

def encode(raw_box, target_height, target_width):
    _, raw_box = cv2.threshold(raw_box, 127, 255, cv2.THRESH_BINARY)
    raw_box = cv2.erode(raw_box, kernel=np.ones((5, 5)), iterations=1)

    raw_box = cv2.resize(raw_box, (target_width, target_height), interpolation=INTERPOLATION_ENCODE)
    _, raw_box = cv2.threshold(raw_box, 127, 255, cv2.THRESH_BINARY)
    return raw_box

def encode_decode(image, intermediate_height, intermediate_width):
    encoded = encode(image, intermediate_height, intermediate_width)
    decoded = decode(encoded)
    print("Base:", np.unique(image, return_counts=True))
    print("Encoded:", np.unique(encoded, return_counts=True))
    print("Decoded:", np.unique(decoded, return_counts=True))
    return {
        "raw_image": image,
        "encoded_image": encoded,
        "decoded_image": decoded,
        "intermediate_height": intermediate_height,
        "intermediate_width": intermediate_width
    }


def display_enc_dec_images(images, intermediate_height, intermediate_width):
    columns, rows = 3, len(images)
    fig = plt.figure()

    # ax enables access to manipulate each of subplots
    ax = []

    for row in range(rows):
        img = images[row]
        transformed = encode_decode(img, intermediate_height, intermediate_width)
        ax.append(fig.add_subplot(rows, columns, columns*row+1))
        plt.imshow(transformed["raw_image"], alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+2))
        plt.imshow(transformed["encoded_image"], alpha=0.25, cmap='gray')

        ax.append(fig.add_subplot(rows, columns, columns*row+3))
        plt.imshow(transformed["decoded_image"], alpha=0.25, cmap='gray')

    plt.colorbar()
    plt.show()


from os.path import join as pjoin
symbol_folder_path = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/"
for file in os.listdir(symbol_folder_path):
    symbol_dataset = np.load(pjoin(symbol_folder_path, file))
    X, y = symbol_dataset['X'], symbol_dataset['y']
    display_enc_dec_images( [X[0], X[1], X[2], X[3], X[4]], 20, 20 )

# plus = np.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/+.npz")
# ones = np.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/1.npz")
# oneX, oneY = ones['X'], ones['y']
# plusX, plusY = plus['X'], plus['y']
# display_enc_dec_images([plusX[0],  plusX[1], plusX[2], plusX[3], plusX[4]], 22, 22)
# display_enc_dec_images([oneX[0],  oneX[1], oneX[2], oneX[3], oneX[4]], 22, 22)
# np.set_printoptions(threshold=np.nan)
# #print(X[0])
# import sys; sys.exit(0)