import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.dataset import save_dataset

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
filename = "checked4b.npz"
background_file_npz = "path/to/folder/with/background/patches/" + filename
full_paper_file_npz = "/path/to/full/paper/screenshots/" + filename

def choose_random_crops(image, display_img, crop_height=45, crop_width=45, crops=50):
    im_height, im_width = image.shape[0], image.shape[1]
    crop_start_h, crop_start_w, crop_end_h, crop_end_w = 50, 50, im_height - 50, im_width - 50
    cv2.rectangle(display_img, (crop_start_w, crop_start_h), (crop_end_w, crop_end_h), color=(255,0,0), thickness=3)

    heights = np.random.randint(crop_start_h, crop_end_h - crop_height, size=crops)
    widths = np.random.randint(crop_start_w, crop_end_w - crop_width, size=crops)
    cropped = [image[heights[i]:heights[i]+crop_height,
               widths[i]:widths[i]+crop_width] for i in range(crops)]

    for i in range(crops):
        cv2.rectangle(display_img, (widths[i], heights[i]), (widths[i]+crop_width, heights[i]+crop_height), (255,0,0), 3)
    return cropped


def show_cropped_images(cropped):
    cols, rows = 5, 5
    fig = plt.figure()
    for i in range(len(cropped)):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(cropped[i], cmap='gray')
    plt.show()


frame_nr = 0
cropped_imgs, imgs = [], []
fps = 25
seconds = 25
while (frame_nr < fps * seconds):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_nr += 1
    if frame_nr % fps != 0:
        continue

    display_img = frame.copy()
    cropped_imgs += choose_random_crops(frame, display_img, 45, 45)

    im_height, im_width = frame.shape[0], frame.shape[1]
    crop_start_h, crop_start_w, crop_end_h, crop_end_w = 50, 50, im_height - 50, im_width - 50
    imgs.append(frame[crop_start_h:crop_end_h, crop_start_w:crop_end_w])
    cv2.imshow('cropped', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cropped_imgs = np.array(cropped_imgs)
y_crop = np.array(["background" for _ in range(len(cropped_imgs))])
save_dataset(cropped_imgs, y_crop, background_file_npz)

imgs = np.array(imgs)
y_img = np.array(["background" for _ in range(len(imgs))])
save_dataset(imgs, y_img, full_paper_file_npz)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
