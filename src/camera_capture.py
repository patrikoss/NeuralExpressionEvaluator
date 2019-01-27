import cv2

from src.utils.expression import get_expressions_boxes
from src.opencv.symbol_classifier import SymbolClassifier
from src.opencv.opencv_frame_handler import OpenCVFrameHandler
from src.sliding_window.sliding_window_frame_handler import SlidingWindowFrameHandler
from src.sliding_window.detector import SymbolDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)



def capture():
    sc1 = SymbolClassifier.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/models/symbol_classifier/s1.obj")
    sw1 = SymbolDetector.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/models/sliding_window/sw1.obj")
    fh = OpenCVFrameHandler(sc1)
    sw = SlidingWindowFrameHandler(sw1)

    while (True):
        # Capture frame-by-frame

        ret, frame = cap.read()
        #symbol_boxes = fh.get_symbols(frame)
        symbol_boxes = sw.get_symbols(frame)
        expressions_boxes = get_expressions_boxes(symbol_boxes, frame)
        sw.show_frames(frame, symbol_boxes, expressions_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


capture()
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()
