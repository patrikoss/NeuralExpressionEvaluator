import cv2
import numpy as np

from src.sliding_window.detector import SymbolDetector
from src.sliding_window.slider import SlidingWindow

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
sc = SymbolDetector.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/models/sliding_window/s19.obj")

def show_detected_symbol(cls_image, color_img, symbol='background', stride=32):
    labels = sc.predict(1./255 * cls_image.reshape(1,*cls_image.shape))
    print(np.sum(labels=='background'))
    symbol_indexes = np.argwhere(labels==symbol)
    for i in range(symbol_indexes.shape[0]):
        _, w_ind, h_ind = symbol_indexes[i]
        #print(h_ind, w_ind, "x")
        height, width = h_ind * stride, w_ind * stride
        cv2.rectangle(color_img, (width, height), (width+32,height+32), (255, 0, 0), 2)
    cv2.imshow("orig image", color_img)


def show_detected_symbol_scaled(image):
    sw = SlidingWindow(image=image, h_starts=1, w_starts=1, h_rescale_factor=0.5, w_rescale_factor=0.5,
                       max_rescales=2, h_window=32, w_window=32, detector=sc)
    symbols = sw.slide()
    for symbol in symbols:
        window_height, window_width = symbol.bottom - symbol.top, symbol.right - symbol.left
        cv2.rectangle(image, (symbol.left, symbol.top), (symbol.left + window_width-1, symbol.top + window_height-1), (255,0,0), 2)
    cv2.imshow('detected symbols: ', image)


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    show_detected_symbol_scaled(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
