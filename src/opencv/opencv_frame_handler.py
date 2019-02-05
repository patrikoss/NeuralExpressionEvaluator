from scipy.ndimage import *
import numpy as np
import cv2
from src.utils.others import INPUT_WIDTH, INPUT_HEIGHT
from src.utils.symbol import SymbolBox, decode_symbol, encode_symbol
from src.utils.base_frame_handler import BaseFrameHandler


class OpenCVFrameHandler(BaseFrameHandler):
    def __init__(self, symbol_classifier):
        self.symbol_classifier = symbol_classifier

    def get_symbols(self, frame):
        """Returns symbol boxes found in the camera frame"""

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        classification_image = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                     cv2.THRESH_BINARY, 7, 2.5)
        localization_image = cv2.erode(classification_image, np.ones((2, 2), np.uint8), iterations=1)

        symbols_positions = self._get_symbols_candidates_location(localization_image)

        # get raw unprocessed symbol boxes
        symbols = (classification_image[top:bottom, left:right] for
                   (top, left, bottom, right) in symbols_positions)
        # encode them to filter only the most relevant features, do not rescale yet
        symbols = [encode_symbol(symbol, None, None, erode=False) for symbol in symbols]
        # decode them to further bring out the most relevant features and rescale
        symbols = [decode_symbol(symbol,INPUT_HEIGHT,INPUT_WIDTH) for symbol in symbols]
        # reshape to fit the classifier format
        symbols = np.array(symbols).reshape(len(symbols), INPUT_HEIGHT, INPUT_WIDTH)
        # make predictions on the symbols
        labels, probs = self.symbol_classifier.predict(symbols)

        symbol_boxes = [SymbolBox(classification_image, top, left, bottom, right, prob, label.decode())
                        for (top, left, bottom, right), prob, label in zip(symbols_positions, probs, labels)]
        return symbol_boxes

    def _get_symbols_candidates_location(self, localization_image):
        """
        :param localization_image: image over which the search is executed.
                                    should be 2D numpy array
        :return: list of rectangles (top,left,bottom,right) specifying potential symbols' locations
        """
        localization_image = 255 - localization_image
        labeled_image, num_features = label(localization_image, np.ones((3, 3), dtype=np.uint8))
        label_indexes, label_counts = np.unique(labeled_image, return_counts=True)
        # TODO if it works too slow filter out the noise at this point
        label_rectangles = np.array(find_objects(labeled_image))

        noise_threshold = 0.01
        valid_blocks_mask = label_counts[1:] > 2 * noise_threshold ** 2 * localization_image.shape[1] * localization_image.shape[0]
        valid_rects = [(slice_vert.start, slice_hor.start, slice_vert.stop - 1, slice_hor.stop - 1)
                       for slice_vert, slice_hor in label_rectangles[valid_blocks_mask]]
        return valid_rects

    def show_frames(self, frame, symbols, expressions):
        """
        Displays the symbols found in the image, as well as additional frame to show how the classifier
        'sees' the symbols candidates
        :param frame: input image
        :param symbols: list of symbolboxes found in the image
        :param expressions: list of expressionboxes found in the image
        :return: None
        """
        super().show_frames(frame, symbols, expressions)

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        decoded_frame = 255 * np.ones((frame_height, frame_width), dtype=np.uint8)
        m = len(symbols)
        for i in range(m):
            symbol = symbols[i]
            if symbol.top < frame_height-INPUT_HEIGHT and symbol.left < frame_width - INPUT_WIDTH:
                encoded_symbol_box = encode_symbol(symbol.get_raw_box(), erode=False)
                encoded_decoded_symbol_box = decode_symbol(encoded_symbol_box, INPUT_HEIGHT, INPUT_WIDTH)
                decoded_frame[symbol.top:symbol.top+INPUT_HEIGHT, symbol.left:symbol.left+INPUT_WIDTH] = encoded_decoded_symbol_box
            cv2.putText(decoded_frame, symbol.prediction_cls,
                        org=(symbol.left, symbol.top),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('classification frame', decoded_frame)
