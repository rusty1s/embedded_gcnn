import tensorflow as tf

from .layer import Layer


class ImageAugment(Layer):
    def __init__(self, **kwargs):
        super(ImageAugment, self).__init__(**kwargs)

    def _call(self, inputs):
        outputs = inputs

        outputs = tf.map_fn(
            lambda image: tf.image.random_flip_left_right(image), outputs)
        outputs = tf.map_fn(
            lambda image: tf.image.random_brightness(image, 0.2), outputs)
        outputs = tf.map_fn(
            lambda image: tf.image.random_contrast(image, 0.8, 1.2), outputs)
        outputs = tf.map_fn(
            lambda image: tf.image.per_image_standardization(image), outputs)
