import numpy as np
from src.utils.others import INPUT_HEIGHT, INPUT_WIDTH
from os.path import join as pjoin
from scipy.misc import imsave


class NoiseGenerator:
    def __init__(self, im_height, im_width, prob):
        self.HEIGHT = im_height
        self.WIDTH = im_width
        self.prob = prob

    def generate(self):
        image = np.random.rand(INPUT_HEIGHT, INPUT_WIDTH) > self.prob
        return 255 * np.array(image, dtype=np.uint8)


def generate_noise_dataset(noise_image_folder):
    images = 500
    base_noise_prob = 0.001
    for prob in range(11):
        ng = NoiseGenerator(INPUT_HEIGHT, INPUT_WIDTH, base_noise_prob*2**prob)
        for i in range(images):
            imgfilename = "noise_{}_{}.jpg".format(prob, i)
            imgfilepath = pjoin(noise_image_folder, imgfilename)
            imsave(imgfilepath, ng.generate())
