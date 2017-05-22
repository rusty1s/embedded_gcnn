from six.moves import xrange

import numpy as np
import tensorflow as tf

from ..tf.convert import sparse_to_tensor


def generate_placeholders(batch_size,
                          levels,
                          num_features,
                          num_labels,
                          has_radians):

    placeholders = {
        'features': [
            tf.placeholder(tf.float32, [None, num_features],
                           'features_{}'.format(i + 1))
            for i in xrange(batch_size)
        ],
        'labels':
        tf.placeholder(tf.int32, [batch_size, num_labels], 'labels'),
        'dropout':
        tf.placeholder(tf.float32, [], 'dropout'),
    }

    adj_name = 'adj_dist' if has_radians else 'adj'
    for j in xrange(1, levels + 1):
        placeholders.update({
            '{}_{}'.format(adj_name, j): [
                tf.sparse_placeholder(
                    tf.float32, name='adj_dist_{}_{}'.format(j, i + 1))
                for i in xrange(batch_size)
            ],
        })

        if has_radians:
            placeholders.update({
                'adj_rad_{}'.format(j): [
                    tf.sparse_placeholder(
                        tf.float32, name='adj_rad_{}_{}'.format(j, i + 1))
                    for i in xrange(batch_size)
                ],
            })

    return placeholders


def feed_dict_with_batch(placeholders, batch, dropout=0.0):
    batch_size = len(batch)
    has_radians = len(batch[0]) == 4
    levels = len(batch[0][1]) - 1
    labels = np.array([batch[i][-1] for i in xrange(batch_size)], np.int32)

    feed_dict = {
        placeholders['labels']: labels,
        placeholders['dropout']: dropout,
    }

    feed_dict.update(
        {placeholders['features'][i]: batch[i][0]
         for i in xrange(batch_size)})

    adj_name = 'adj_dist' if has_radians else 'adj'
    for j in xrange(levels):
        feed_dict.update({
            placeholders['{}_{}'.format(adj_name, j + 1)][i]: sparse_to_tensor(batch[i][1][j])
            for i in xrange(batch_size)
        })

        if has_radians:
            feed_dict.update({
                placeholders['adj_rad_{}'.format(j + 1)][i]: sparse_to_tensor(batch[i][2][j])
                for i in xrange(batch_size)
            })

    return feed_dict
