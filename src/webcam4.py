import numpy as np
import cv2
from denoise_mask import *
from groups import *
from scipy.ndimage import *
from symbol_localization import get_symbols_candidates_location

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)

while (True):
    # Capture frame-by-frame
    # for i in range(30):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 7, 2.5)
    mask = cv2.filter2D(threshold, -1, denoise_mask(7, 1))
    # ret, threshold=cv2.threshold(threshold*(1-mask), 120,255,cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    # d_im = cv2.dilate(threshold, kernel, iterations=1)
    e_im = cv2.erode(threshold, kernel, iterations=1)
    #print(e_im)

    cv2.imshow("aa", e_im)
    for bounding_box in get_symbols_candidates_location(e_im):
        cv2.rectangle(frame, (bounding_box[1], bounding_box[0]), (bounding_box[3], bounding_box[2]), (0, 255, 0), 3)

    # cv2.imshow("with threshold", threshold)

    # cv2.imshow('with threshold', threshold)

    cv2.imshow("with threshold", threshold)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
