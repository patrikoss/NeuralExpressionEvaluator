import itertools
import cv2
import numpy as np
from src.utils.symbol import SymbolBox


class SlidingWindow:
    def __init__(self, image, h_starts, w_starts, detector_window_h, detector_window_w, detector, windows):
        """
        :param image: input image
        :param h_starts: number of starting positions for a window withing the window height.
                        Example: if a window has a height of 30, and h_starts = 3 then
                        the window is going to be offset in y-coordinate by
                        0pixels, 10pixels and 20pixels
        :param w_starts: number of starting positions for a window withing the window width.
                        Example: if a window has a width of 30, and w_starts = 3 then
                        the window is going to be offset in x-coordintate by
                        0pixels, 10pixels and 20pixels
        :param detector_window_h: the receptive field in width of the window. Should be
                        derived based on the architecture.
                        Example: MP(2,2) -> MP(2,2) -> MP(2,2), where MP - max-pooling
                        The receptive field is (8,8) and so 8 should be passed as detector_window_h
        :param detector_window_w: the receptive field in height of the window. Should be
                        derived based on the architecture.
                        Example: MP(2,2) -> MP(2,2) -> MP(2,2), where MP - max-pooling
                        The receptive field is (8,8) and so 8 should be passed as detector_window_w
        :param detector: instance of class SymbolDetector - should be already trained
        :param windows: list of window shapes to apply to image in a format (height, width)
                        Example: [(32,32), (64,32)]

        """
        self.image = image

        self.im_height, self.im_width = image.shape[0], image.shape[1]

        h_step, w_step = detector_window_h // h_starts, detector_window_w // w_starts
        self.offsets = list(itertools.product(range(0,detector_window_h, h_step), range(0,detector_window_w,w_step)))

        self.h_window, self.w_window = detector_window_h, detector_window_w

        # instead of changing the window size when sliding, we can resize the image so that a
        # detectr_window_h, detector_window_w window corresponds to the appropriate window
        # Enlarging the window corresponds to making the image smaller
        self.image_rescales = [(detector_window_h/win_height, detector_window_w/win_width)
                         for win_height, win_width in windows]

        self.symbols = list()
        self.detector = detector

        self.background_cls_index = np.where(self.detector.label_binarizer.classes_ == 'background')[0][0]

    def filter_out_unsure(self, threshold=0.5):
        """
        Filters out the boxes with lower confidence score than specified
        """
        self.symbols = [symbol for symbol in self.symbols if symbol.prediction_confidence > threshold]

    def suppress(self, iou_threshold=0.01):
        """
        Removes those predictions that the detector was unsure of
        """
        def iou(symbol1, symbol2):
            left = max(symbol1.left, symbol2.left)
            right = min(symbol1.right, symbol2.right)
            top = max(symbol1.top, symbol2.top)
            bottom = min(symbol1.bottom, symbol2.bottom)
            intersection = max(right - left, 0) * max(bottom - top, 0)
            s1_area = (symbol1.right-symbol1.left)*(symbol1.bottom-symbol1.top)
            s2_area = (symbol2.right-symbol2.left)*(symbol2.bottom-symbol2.top)
            union = s1_area + s2_area - intersection
            return intersection / union

        candidates = sorted([(symbol.prediction_confidence, dummy_ind, symbol) for dummy_ind, symbol in enumerate(self.symbols)])
        valid = []
        for _, _, candidate in candidates:
            for symbol in valid:
                if iou(symbol, candidate) >= iou_threshold:
                    break
            else:
                valid.append(candidate)
        self.symbols = valid

    def find_symbol_boxes(self, rescale_h, rescale_w):
        """
        Returns list of candidates symbols found in a given image
        :param rescale_h: height rescale factor of the image(necessary for variable shaped windows)
        :param rescale_w: width rescale factor of the image(necessary for variable shaped windows)
        """
        target_h = int(self.im_height * rescale_h)
        target_w = int(self.im_width * rescale_w)

        # if the window size exceeds image size, break
        if target_h < self.h_window or target_w < self.w_window:
            return []

        symbols = list()

        image = cv2.resize(self.image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        for offset_h, offset_w in self.offsets:
            offset_img = image[offset_h:, offset_w:, :]
            offset_img = offset_img.reshape(1, *offset_img.shape)

            preds_labels, preds_conf = self.detector.predict(offset_img)
            symbol_windows_indexes = np.argwhere(preds_labels!='background')

            for ex, win_h_ind, win_w_ind in symbol_windows_indexes:
                top = offset_h + self.h_window * win_h_ind
                left = offset_w + self.w_window * win_w_ind
                bottom = top + self.h_window
                right = left + self.w_window

                top, bottom = int(top * 1.0/rescale_h), int(bottom * 1.0/rescale_h)
                left, right = int(left * 1.0/rescale_w), int(right * 1.0/rescale_w)
                symbols.append(
                    SymbolBox(image=self.image, top=top, left=left, bottom=bottom,right=right,
                              prediction_cls=str(preds_labels[ex,win_h_ind,win_w_ind]),
                              prediction_confidence=preds_conf[ex,win_h_ind,win_w_ind])
                )
        return symbols

    def slide(self):
        """
        Returns symbols found in the image
        """
        for rescale_h, rescale_w in self.image_rescales:
            self.symbols += self.find_symbol_boxes(rescale_h, rescale_w)

        # filter out predictions for which we do not have at least given confidence
        self.filter_out_unsure(threshold=0.3)

        # suppress some of the overlapping boxes
        self.suppress(iou_threshold=0.01)

        return self.symbols





