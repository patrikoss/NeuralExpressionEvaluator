import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.loader import save_dataset

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
background_file_npz = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/raw_symbols/npz_symbols/background.npz"

def choose_random_crops(image, crop_height=45, crop_width=45, crops=50):
    im_height, im_width = image.shape[0], image.shape[1]
    heights = np.random.randint(0, im_height - crop_height, size=crops)
    widths = np.random.randint(0, im_width - crop_width, size=crops)
    cropped = [image[heights[i]:heights[i]+crop_height,
               widths[i]:widths[i]+crop_width] for i in range(crops)]
    return cropped


def show_cropped_images(cropped):
    cols, rows = 5, 5
    fig = plt.figure()
    for i in range(len(cropped)):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(cropped[i], cmap='gray')
    plt.show()


frame_nr = 0
cropped_imgs = []
while (frame_nr < 30 * 50):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_nr += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_nr % 30 != 0:
        continue
    classification_image = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 7, 2.5)
    #cv2.imshow('grayframe', gray_frame)
    cv2.imshow('cls image', classification_image)

    cropped_imgs += choose_random_crops(classification_image, 45, 45)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cropped_imgs = np.array(cropped_imgs)
y = np.array(["background" for _ in range(len(cropped_imgs))])
save_dataset(cropped_imgs, y, background_file_npz)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
