from scipy.ndimage import *
import numpy as np


def get_symbols_candidates_location(image):
    image = 255 - image
    labeled_image, num_features = label(image, np.ones((3, 3), dtype=np.uint8))  # , np.ones((3, 3), dtype=np.uint8))
    label_indexes, label_counts = np.unique(labeled_image, return_counts=True)
    # TODO if it works too slow filter out the noise at this point
    label_rectangles = np.array(find_objects(labeled_image))

    noise_threshold = 0.01
    valid_blocks_mask = label_counts[1:] > noise_threshold ** 2 * image.shape[1] * image.shape[0]
    valid_rects = [(slice_vert.start, slice_hor.start, slice_vert.stop - 1, slice_hor.stop - 1)
                   for slice_vert, slice_hor in label_rectangles[valid_blocks_mask]]
    return valid_rects
