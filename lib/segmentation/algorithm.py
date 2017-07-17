import numpy as np
from skimage import color, segmentation


def slic(image, num_segments, compactness=10, max_iterations=10, sigma=0):
    image = _preprocess(image)
    return segmentation.slic(image, num_segments, compactness, max_iterations,
                             sigma)


def slic_fixed(num_segments, compactness=10, max_iterations=10, sigma=0):
    def slic_image(image):
        return slic(image, num_segments, compactness, max_iterations, sigma)

    return slic_image


def quickshift(image, ratio=1, kernel_size=5, max_dist=1, sigma=0):
    image = _preprocess(image)
    return segmentation.quickshift(
        image, ratio, kernel_size, max_dist, sigma=sigma)


def quickshift_fixed(ratio=1, kernel_size=5, max_dist=1, sigma=0):
    def quickshift_image(image):
        return quickshift(image, ratio, kernel_size, max_dist, sigma)

    return quickshift_image


def felzenszwalb(image, scale=1, min_size=1, sigma=0):
    image = _preprocess(image)
    return segmentation.felzenszwalb(image, scale, sigma, min_size)


def felzenszwalb_fixed(scale=1, min_size=1, sigma=0):
    def felzenszwalb_image(image):
        return felzenszwalb(image, scale, min_size, sigma)

    return felzenszwalb_image


def _preprocess(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return color.gray2rgb(
            np.reshape(image, (image.shape[0], image.shape[1])))
    else:
        return image
