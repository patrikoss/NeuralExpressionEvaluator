from PIL import Image
import matplotlib.pyplot as plt
import cv2

SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SYMBOLS += ["-", "+", "times", "forward_slash"]

SEED = 1

CATEGORIES = len(SYMBOLS)
INPUT_HEIGHT = 45
INPUT_WIDTH = 45
INPUT_CHANNEL = 1


def show_image(np_array):
    """
    Displays the array as an image
    """
    Image.fromarray(np_array).show()


def show_images(images, labels, bgr=True):
    """
    Draws a set of images on a plane and display them.
    :param images: numpy list of images
    :param labels: labels for each image
    :param bgr: True if the images are in bgr format. False if it is in rgb
    """
    columns, rows = 5, len(images) // 5 + 1
    fig = plt.figure()

    # ax enables access to manipulate each of subplots
    axes = []

    for row in range(rows):
        for col in range(columns):
            image_index = columns*row + col
            if image_index >= images.shape[0]:
                break

            img = images[image_index]
            if bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = fig.add_subplot(rows, columns, image_index+1)
            ax.title.set_text(str(labels[image_index]))

            axes.append(ax)
            plt.imshow(img)

    plt.colorbar()
    plt.show()
