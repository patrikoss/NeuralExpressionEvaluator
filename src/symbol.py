from scipy.ndimage import *
import numpy as np
import cv2
from src.utils import INPUT_WIDTH, INPUT_HEIGHT


def decode_symbol(encoded_symbol_box, rescaled_height=INPUT_HEIGHT, rescaled_width=INPUT_WIDTH):
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


def encode_symbol(raw_symbol_box, encoded_height=None, encoded_width=None):
    _, encoded_box = cv2.threshold(raw_symbol_box, 127, 255, cv2.THRESH_BINARY)
    if encoded_height is not None and encoded_width is not None:
        encoded_box = cv2.resize(encoded_box, (encoded_width, encoded_height), interpolation=cv2.INTER_LINEAR)
        _, encoded_box = cv2.threshold(encoded_box, 127, 255, cv2.THRESH_BINARY)
    return encoded_box


class SymbolBox:
    def __init__(self, image, top, left, bottom, right):
        self.top, self.left, self.bottom, self.right = top, left, bottom, right
        self.center_row, self.center_col = (top+bottom)//2, (left+right)//2
        self.image = image

    def get_raw_box(self):
        return self.image[self.top: self.bottom+1, self.left:self.right+1]

    def decode_box(self, rescaled_height=INPUT_HEIGHT, rescaled_width=INPUT_WIDTH):
        encoded_box = self.encode_box()
        return decode_symbol(encoded_box, rescaled_height, rescaled_width)

    def encode_box(self, encoded_height=None, encoded_width=None):
        raw_symbol_box = self.get_raw_box()
        return encode_symbol(raw_symbol_box, encoded_height, encoded_width)


def get_symbols_candidates_location(localization_image, classification_image):
    localization_image = 255 - localization_image
    labeled_image, num_features = label(localization_image, np.ones((3, 3), dtype=np.uint8))
    label_indexes, label_counts = np.unique(labeled_image, return_counts=True)
    # TODO if it works too slow filter out the noise at this point
    label_rectangles = np.array(find_objects(labeled_image))

    noise_threshold = 0.01
    valid_blocks_mask = label_counts[1:] > 2 * noise_threshold ** 2 * localization_image.shape[1] * localization_image.shape[0]
    valid_rects = [SymbolBox(classification_image, slice_vert.start, slice_hor.start, slice_vert.stop - 1, slice_hor.stop - 1)
                   for slice_vert, slice_hor in label_rectangles[valid_blocks_mask]]
    return valid_rects
