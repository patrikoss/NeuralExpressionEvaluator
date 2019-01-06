import cv2
import numpy as np
import os
from os.path import join as pjoin
from .loader import save_dataset
from . import symbol_localization as sl

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
    # print("Base:", np.unique(image, return_counts=True))
    # print("Encoded:", np.unique(encoded, return_counts=True))
    # print("Decoded:", np.unique(decoded, return_counts=True))
    return {
        "raw_image": image,
        "encoded_image": encoded,
        "decoded_image": decoded,
        "intermediate_height": intermediate_height,
        "intermediate_width": intermediate_width
    }

def generate_encoded_decoded_dataset(source_folder_path,target_folder_path,
                                     intermediate_shape):

    for filename in os.listdir(source_folder_path):
        symbol_name = filename[:-4]
        source_dataset = np.load(pjoin(source_folder_path, filename))
        X, y = source_dataset['X'], source_dataset['y']
        print("Resizing symbol: {}".format(symbol_name))
        count = 0
        newX, newY = [], []
        for example in range(X.shape[0]):
            symbol = X[example]

            # append self
            newX.append(X[0]); newY.append(symbol_name)

            for intermediate_height, intermediate_width in intermediate_shape:
                encoded = sl.encode_symbol(symbol, intermediate_height, intermediate_width)
                decoded = sl.decode_symbol(encoded, 45, 45)
                newX.append(decoded); newY.append(symbol_name)

            count += 1
            if count % 100 == 0:
                print("Transformed {} images for symbol {}".format(count, symbol_name))

        newX = np.array(newX)
        newY = np.array(newY, dtype='|S6')  # data type: string
        print(newX.shape, newY.shape, pjoin(target_folder_path, filename))
        save_dataset(newX, newY, pjoin(target_folder_path, filename))


# source_folder_path = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/"
# target_folder_path = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/encoded_decoded_npz_symbols"
# intermediate_height = 20
# intermediate_width = 20
# generate_encoded_decoded_dataset(source_folder_path, target_folder_path,
#                                  intermediate_shape=[(70,70), (20,20), (30,20), (40,20), (70,40)])
