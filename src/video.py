import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#import ipdb; ipdb.set_trace()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray0 = cv2.GaussianBlur(gray0, (5, 5), 0)

    kernel = np.ones((3, 3), np.float32) / 9
    gray1 = cv2.filter2D(gray0, -1, kernel)


    gray2 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #img = cv2.resize(gray, (640,120))

    # Display the resulting frame
    cv2.imshow('frame', gray2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()