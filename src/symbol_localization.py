from scipy.ndimage import *
import numpy as np
import cv2


class SymbolBox:
    def __init__(self, image, top, left, bottom, right):
        self.top, self.left, self.bottom, self.right = top, left, bottom, right
        self.center_row, self.center_col = (top+bottom)//2, (left+right)//2
        self.image = image

    def get_raw_box(self):
        return self.image[self.top: self.bottom+1, self.left:self.right+1]

    def get_rescaled_box(self, rescaled_height=45, rescaled_width=45):
        raw_box = self.get_raw_box()
        raw_height, raw_width = raw_box.shape
        difference = abs(raw_height - raw_width)
        if raw_height < raw_width:
            padding = (((difference+1)//2, difference//2), (0, 0))
        else:
            padding = ((0, 0), ((difference + 1) // 2, difference // 2))
        box = np.pad(raw_box, padding, mode='constant', constant_values=255)
        return cv2.resize(box, (rescaled_height, rescaled_width), interpolation=cv2.INTER_NEAREST)


def get_symbols_candidates_location(image):
    image = 255 - image
    labeled_image, num_features = label(image, np.ones((3, 3), dtype=np.uint8))  # , np.ones((3, 3), dtype=np.uint8))
    label_indexes, label_counts = np.unique(labeled_image, return_counts=True)
    # TODO if it works too slow filter out the noise at this point
    label_rectangles = np.array(find_objects(labeled_image))

    noise_threshold = 0.01
    valid_blocks_mask = label_counts[1:] > noise_threshold ** 2 * image.shape[1] * image.shape[0]
    valid_rects = [SymbolBox(image, slice_vert.start, slice_hor.start, slice_vert.stop - 1, slice_hor.stop - 1)
                   for slice_vert, slice_hor in label_rectangles[valid_blocks_mask]]
    return valid_rects
