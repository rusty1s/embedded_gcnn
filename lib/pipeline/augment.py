import random
import numpy as np
from numpy import pi as PI


def augment_batch(batch):
    batch_augmented = []
    for example in batch:
        features, adjs_dist, adjs_rad, label = example

        features = random_brightness(features, 0, 3, max_delta=0.1)
        features = random_contrast(features, 0, 3, max_delta=0.2)

        adjs_rad = random_flip_left_right_adjs(adjs_rad)

        batch_augmented.append((features, adjs_dist, adjs_rad, label))

    return batch_augmented


def flip_left_right_adj(adj_rad):
    adj_rad.data = 2 * PI - adj_rad.data
    return adj_rad


def random_flip_left_right_adjs(adjs_rad):
    if bool(random.getrandbits(1)):
        adjs_rad = [flip_left_right_adj(adj_rad) for adj_rad in adjs_rad]

    return adjs_rad


def adjust_brightness(features, start_idx, end_idx, delta):
    colors = features[:, start_idx:end_idx]
    colors = colors + delta
    colors = np.clip(colors, 0, 1)

    return np.concatenate(
        (features[:, :start_idx], colors, features[:, end_idx:]), axis=1)


def random_brightness(features, start_idx, end_idx, max_delta):
    rand = random.uniform(-max_delta, max_delta)
    return adjust_brightness(features, start_idx, end_idx, rand)


def adjust_contrast(features, start_idx, end_idx, delta):
    colors = features[:, start_idx:end_idx]
    mean = colors.mean(axis=0)
    colors = (colors - mean) * (1 + delta) + mean
    colors = np.clip(colors, 0, 1)

    return np.concatenate(
        (features[:, :start_idx], colors, features[:, end_idx:]), axis=1)


def random_contrast(features, start_idx, end_idx, max_delta):
    rand = random.uniform(-max_delta, max_delta)
    return adjust_contrast(features, start_idx, end_idx, rand)
