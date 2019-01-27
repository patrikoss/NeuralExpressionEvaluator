import itertools
import cv2
import numpy as np
from src.utils.symbol import SymbolBox

class SlidingWindow:
    def __init__(self, image, h_starts, w_starts, h_rescale_factor, w_rescale_factor, max_rescales,
                 h_window, w_window, detector):
        """
        :param image: input image
        """
        self.image = image

        self.im_height, self.im_width = image.shape[0], image.shape[1]

        h_step, w_step = h_window // h_starts, w_window // w_starts
        self.offsets = list(itertools.product(range(0,h_window, h_step), range(0,w_window,w_step)))

        self.h_window, self.w_window = h_window, w_window

        self.h_rescale_factor, self.w_rescale_factor = h_rescale_factor, w_rescale_factor
        self.max_rescales = max_rescales

        self.symbols = list()
        self.detector = detector

        self.background_cls_index = np.where(self.detector.label_binarizer.classes_ == 'background')[0][0]

    def filter_out_unsure(self, threshold=0.5):
        """
        Filters out the boxes with lower confidence score than specified
        """
        self.symbols = [symbol for symbol in self.symbols if symbol.prediction_confidence > threshold]

    def suppress(self):
        """
        :param boxes:
        :return:
        """
        pass

    def find_symbol_boxes(self, rescale_h, rescale_w):
        """
        :param h_rescale:
        :param w_rescale:
        :param h_offset:
        :param w_offset:
        :return: dict mapping boxes(start_h, start_w, box_h, box_w) to symbol classess along with the confidence
        """
        target_h = int(self.im_height * rescale_h)
        target_w = int(self.im_width * rescale_w)

        # if the window size exceeds image size, break
        if target_h < self.h_window or target_w < self.w_window:
            return []

        symbols = list()

        image = cv2.resize(self.image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        image = 1./255 * image

        for offset_h, offset_w in self.offsets:
            offset_img = image[offset_h:, offset_w:, :]
            offset_img = offset_img.reshape(1, *offset_img.shape)

            preds_labels, preds_conf = self.detector.predict(offset_img)
            print(preds_labels)
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
                              prediction_cls=preds_labels[ex,win_h_ind,win_w_ind],
                              prediction_confidence=preds_conf[ex,win_h_ind,win_w_ind])
                )
        return symbols




    def slide(self):
        """
        :return: dict mapping boxes(start_h, start_w, box_h, box_w) to symbol classess along with the confidence
        """
        rescale_h, rescale_w = 1, 1

        for rescale_nr in range(self.max_rescales):
            self.symbols += self.find_symbol_boxes(rescale_h, rescale_w)
            rescale_h *= self.h_rescale_factor
            rescale_w *= self.w_rescale_factor

        # filter out predictions for which we do not have at least 0.5 confidence
        self.filter_out_unsure(threshold=0.5)

        # suppress some of the overlapping boxes
        self.suppress()

        return self.symbols





