import numpy as np
import cv2
from src.utils.others import INPUT_WIDTH, INPUT_HEIGHT


def decode_symbol(encoded_symbol_box, rescaled_height=INPUT_HEIGHT, rescaled_width=INPUT_WIDTH):
    """
    Transforms back the encoded image into the target shape.
    :return: decoded symbol image
    """
    encoded_box = encoded_symbol_box
    raw_height, raw_width = encoded_box.shape
    difference = abs(raw_height - raw_width)
    if raw_height < raw_width:
        padding = (((difference + 1) // 2, difference // 2), (0, 0))
    else:
        padding = ((0, 0), ((difference + 1) // 2, difference // 2))

    decoded_box = np.pad(encoded_box, padding, mode='constant', constant_values=255)
    _, decoded_box = cv2.threshold(decoded_box, 127, 255, cv2.THRESH_BINARY)
    decoded_box = cv2.resize(decoded_box, (rescaled_width, rescaled_height), interpolation=cv2.INTER_LINEAR)
    _, decoded_box = cv2.threshold(decoded_box, 127, 255, cv2.THRESH_BINARY)
    return decoded_box


def encode_symbol(raw_symbol_box, encoded_height=None, encoded_width=None, erode=False):
    """
    Transforms the input symbol image into intermediate shape, so that it can be
    decoded later. This creates realistic distortions of the image
    :return: encoded symbol image
    """
    _, encoded_box = cv2.threshold(raw_symbol_box, 127, 255, cv2.THRESH_BINARY)
    if erode:
        encoded_box = cv2.erode(encoded_box, kernel=np.ones((3, 3)), iterations=1) # TODO: necessary?
    if encoded_height is not None and encoded_width is not None:
        encoded_box = cv2.resize(encoded_box, (encoded_width, encoded_height), interpolation=cv2.INTER_LINEAR)
        _, encoded_box = cv2.threshold(encoded_box, 127, 255, cv2.THRESH_BINARY)
    return encoded_box


class SymbolBox:
    """
    Class representing a symbol found in the image at the specified location
    """

    def __init__(self, image, top, left, bottom, right, prediction_confidence, prediction_cls):
        """
        :param image: image in which the symbol has been found
        :param top: top y-coordinate of the symbol found(notice that top <= bottom)
        :param left: left x-coordinate of the symbol found
        :param top: bottom y-coordinate of the symbol(notice that top <= bottom)
        :param left: right x-coordinate of the symbol
        :param prediction_confidence: confidence of the classifier as to the symbol class
        :param prediction_cls: predicted symbol class
        """
        self.top, self.left, self.bottom, self.right = top, left, bottom, right
        self.center_row, self.center_col = (top+bottom)//2, (left+right)//2
        self.image = image
        self.prediction_confidence = prediction_confidence
        self.prediction_cls = prediction_cls

    def get_raw_box(self):
        """
        Returns the section of the image corresponding to the symbol
        """
        return self.image[self.top: self.bottom+1, self.left:self.right+1]

