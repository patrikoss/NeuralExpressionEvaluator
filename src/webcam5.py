import numpy as np
import cv2

from src.symbol_localization import get_symbols_candidates_location
from src.expression_localization import get_expressions_boxes
from src.symbol_classifier import SymbolClassifier1
from src.utils import show_image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
sc1 = SymbolClassifier1.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/models/symbol_classifier1/sc4_enc_dec.obj")
#sc1 = SymbolClassifier1.load("/home/patryk/PycharmProjects/MathExpressionEvaluator/models/symbol_classifier1/sc1.obj")

def show_decoded_frame(symbols, symbols_preds):
    decoded_frame = 255 * np.ones((480,640), dtype=np.uint8)
    m = len(symbols)
    for i in range(m):
        symbol = symbols[i]
        if symbol.top < 480-45 and symbol.left < 640 - 45:
            decoded_frame[symbol.top:symbol.top+45, symbol.left:symbol.left+45] = symbol.decode_box()
        cv2.putText(decoded_frame, symbols_preds[i].decode(), org=(symbol.left, symbol.top), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow('decoded frame', decoded_frame)


def show_symbols_and_expression_location(image, classification_image, symbols, symbols_preds):
    for i, symbol in enumerate(symbols):
        cv2.rectangle(image, (symbol.left, symbol.top), (symbol.right, symbol.bottom), (0, 255, 0), 3)
        cv2.putText(image, symbols_preds[i].decode(), org=(symbol.left, symbol.top), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,255,0), thickness=2)
    show_decoded_frame(symbols, symbols_preds)

    expression_boxes = get_expressions_boxes(symbols, classification_image)
    for exp_box in expression_boxes:
        cv2.rectangle(image, (exp_box.left, exp_box.top), (exp_box.right, exp_box.bottom), (255, 0, 0), 1)

    cv2.imshow("original image", image)


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    classification_image = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 7, 2.5)
    localization_image = cv2.erode(classification_image, np.ones((2, 2), np.uint8), iterations=1)


    symbols = get_symbols_candidates_location(localization_image, classification_image)
    symbols_raw_boxes = np.array([symbol.decode_box() for symbol in symbols]).reshape(-1,45,45)
    symbols_preds = sc1.predict(symbols_raw_boxes)


    show_decoded_frame(symbols, symbols_preds)
    show_symbols_and_expression_location(frame,classification_image,symbols, symbols_preds)


    cv2.imshow("localization_image", localization_image)
    cv2.imshow("classification image", classification_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
