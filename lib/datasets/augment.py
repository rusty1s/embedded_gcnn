import random
import numpy as np


def flip_left_right_image(image):
    return np.fliplr(image)


def random_flip_left_right_image(image, rand=None):
    rand = bool(random.getrandbits(1)) if rand is None else rand
    return flip_left_right_image(image) if rand else image


def adjust_brightness(image, delta):
    image = image + delta
    image = np.clip(image, 0, 1)
    return image


def random_brightness(image, max_delta):
    rand = random.uniform(-max_delta, max_delta)
    return adjust_brightness(image, rand)


def adjust_contrast(image, delta):
    mean = image.mean(axis=(0, 1))
    image = (image - mean) * (1 + delta) + mean
    image = np.clip(image, 0, 1)
    return image


def random_contrast(image, max_delta):
    rand = random.uniform(-max_delta, max_delta)
    return adjust_contrast(image, rand)
