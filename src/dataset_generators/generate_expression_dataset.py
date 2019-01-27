from src.utils.others import SEED, INPUT_HEIGHT, INPUT_WIDTH
from scipy.ndimage import zoom
from os.path import join as pjoin
import numpy as np
import cv2
import os

np.random.seed(SEED)

class TransformedSymbolGenerator:
    def __init__(self, symbol_dataset_npz):
        symbol_dataset = np.load(symbol_dataset_npz)
        self.X = symbol_dataset['X']
        self.m = symbol_dataset['X'].shape[0]

    def bolden(self, np_image):
        kernel_size = np.random.randint(1,3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bolden_image = cv2.erode(np_image, kernel, iterations=1)
        return bolden_image

    def zoom(self, np_image):
        mean, stddev = 1, 0.1
        scale_factor = max(np.random.normal(mean, stddev), 0.1)
        return zoom(np_image, zoom=scale_factor)

    def generate(self):
        image_nr = np.random.randint(0, self.m)
        image = self.X[image_nr]
        image = self.bolden(image)
        image = self.zoom(image)
        return image


class ExpressionGenerator:
    def __init__(self, image_height, image_width, symbols_dataset_npz_folder):
        self.HEIGHT = image_height
        self.WIDTH = image_width

        # initialize symbol generators for all symbols
        self.tsg = dict()
        for symbol_npz_filename in os.listdir(symbols_dataset_npz_folder):
            symbol_npz_filepath = pjoin(symbols_dataset_npz_folder, symbol_npz_filename)
            symbol = symbol_npz_filename[:-4]  # cut off .npz extension
            self.tsg[symbol] = TransformedSymbolGenerator(symbol_npz_filepath)

    def generate(self):

        def paste_symbol_into_image(symbol, image, height, width):
            """Returns true if the symbol has been pasted into image"""
            symbol_height, symbol_width = symbol.shape
            if height + symbol_height >= self.HEIGHT:
                return False
            if width + symbol_width >= self.WIDTH:
                return False
            image[height:height+symbol_height, width:width+symbol_width] = symbol
            return True

        pasted = list()

        symbols = list(self.tsg.keys())
        image = 255 * np.ones((self.HEIGHT, self.WIDTH), np.uint8)
        height = self.HEIGHT // 10
        while height < self.HEIGHT:
            width = self.WIDTH // 10
            while width < self.WIDTH:
                symbol = self.tsg[symbols[np.random.randint(0, len(symbols))]].generate()
                symbol_height, symbol_width = symbol.shape
                if paste_symbol_into_image(symbol, image, height, width):
                    pasted.append((symbol,
                         height + symbol_height//2, width + symbol_width//2,  # symbol center coordinates
                         symbol_height, symbol_width))
                width += symbol_width + max(3, int(np.random.normal(15, 3)))
            height += symbol_height + max(10, int(np.random.normal(50, 5)))
        return image, pasted
