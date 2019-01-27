import numpy as np
from src.utils.dataset import load_dataset_by_symbol, load_dataset, shuffle_dataset, rescale_dataset
from src.utils.others import show_images
import cv2

def preprocess_symbol(img, target_channel_size):
    assert len(img.shape) == 2
    img = cv2.erode(img, kernel=np.ones((3, 3)), iterations=1)
    if target_channel_size == 2:
        pass
    elif target_channel_size == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def preprocess_background(img):
    assert len(img.shape) in {2,3}
    return img

def merge_symbols_and_backgrounds(symbols, backgrounds):
    return np.minimum(symbols, backgrounds)

def generate_gray_symbols_with_background(symbols_folder, background_folder, batch_per_class,
                                          label_binarizer, target_height=32, target_width=32):
    """Batch size should be a multiple of #symbol classess"""
    symbols, background = load_dataset_by_symbol(symbols_folder), load_dataset(background_folder)
    for s in symbols:
        s_X, s_y = symbols[s]
        new_s_X = []
        for i in range(s_X.shape[0]):
            new_s_X.append(preprocess_symbol(s_X[i], 3))
        symbols[s] = np.array(new_s_X), s_y

    bg_X, bg_y = background['X'], background['y']
    new_bg_X = []
    for i in range(bg_X.shape[0]):
        new_bg_X.append(preprocess_background(bg_X[i]))
    bg_X = np.array(bg_X)

    # Image format
    # bg_X, bg_y
    # symbols[symbol_name -> symbol_X, symbol_y]

    label_binarizer.fit(np.array([s for s in symbols] + ["background"]))
    batch_nr = 0
    while True:
        batch_X, batch_y = [], []
        batch_bg = bg_X[np.random.randint(0, bg_X.shape[0], batch_per_class * len(symbols))]
        for s in symbols:
            symbol_X, symbol_y = symbols[s]
            m = symbol_X.shape[0]
            start_ind = (batch_nr * batch_per_class) % m
            end_ind = start_ind + batch_per_class
            if end_ind <= m:
                batch_X.append(symbol_X[start_ind:end_ind])
            else:
                batch_X.append(symbol_X[start_ind:])
                batch_X.append(symbol_X[0:end_ind % m])
            batch_y.append(symbol_y[0:batch_per_class])
        batch_X, batch_y = np.concatenate(batch_X), np.concatenate(batch_y)

        batch_X = merge_symbols_and_backgrounds(batch_X, batch_bg)

        # add background
        random_bg_ind = np.random.randint(0,bg_X.shape[0], batch_per_class*len(symbols))
        bg_batch_x, bg_batch_y = bg_X[random_bg_ind], bg_y[random_bg_ind]

        batch_X, batch_y = np.concatenate([batch_X, bg_batch_x]), np.concatenate([batch_y, bg_batch_y])


        # one hot encode y vector
        batch_y = label_binarizer.transform(batch_y)

        # reshape to fit the keras format
        #batch_X = batch_X.reshape(*batch_X.shape)
        batch_y = batch_y.reshape(batch_y.shape[0],1,1,batch_y.shape[1])

        # rescale accordingly
        batch_X = rescale_dataset(batch_X, target_height, target_width)



        yield shuffle_dataset(1./255 * batch_X, batch_y)

        batch_nr += 1



