import cv2
import os
import sys

root_path = os.path.dirname(os.path.realpath(__file__)).rsplit("/", 1)[0]
sys.path.extend([root_path])

from src.utils.expression import get_expressions_boxes
from src.opencv.symbol_classifier import SymbolClassifier
from src.opencv.opencv_frame_handler import OpenCVFrameHandler
from src.sliding_window.sliding_window_frame_handler import SlidingWindowFrameHandler
from src.sliding_window.detector import SymbolDetector


sw_model = os.path.join(root_path, "models", "sliding_window", "sw14.obj")
opencv_model = os.path.join(root_path, "models", "symbol_classifier", "sc_final.obj")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)


def init_opencv_detector(modelpath):
    sc1 = SymbolClassifier.load(modelpath)
    fh = OpenCVFrameHandler(sc1)
    return fh


def init_sliding_window_detector(modelpath):
    sd1 = SymbolDetector.load(modelpath)
    sw = SlidingWindowFrameHandler(sd1)
    return sw


def capture(method):
    if method == 'opencv':
        fh = init_opencv_detector(opencv_model)
    elif method == 'sliding_window':
        fh = init_sliding_window_detector(sw_model)
    else:
        raise Exception("Unknown method")

    while (True):
        # Capture frame-by-frame

        ret, frame = cap.read()
        symbol_boxes = fh.get_symbols(frame)
        expressions_boxes = get_expressions_boxes(symbol_boxes, frame)
        fh.show_frames(frame, symbol_boxes, expressions_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture(method='opencv')
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()
