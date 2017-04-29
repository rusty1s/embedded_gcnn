from six.moves import xrange

import numpy as np
import tensorflow as tf


def embedded_placeholders(batch_size, levels, num_features, num_labels):
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

    for j in xrange(1, levels + 1):
        placeholders.update({
            'adj_dist_{}'.format(j): [
                tf.sparse_placeholder(
                    tf.float32, name='adj_dist_{}_{}'.format(j, i + 1))
                for i in xrange(batch_size)
            ],
            'adj_rad_{}'.format(j): [
                tf.sparse_placeholder(
                    tf.float32, name='adj_rad_{}_{}'.format(j, i + 1))
                for i in xrange(batch_size)
            ],
        })

    return placeholders


def embedded_feed_dict(placeholders, batch, dropout=0.0):
    batch_size = len(batch)
    levels = len(batch[0][1]) - 1
    labels = np.array([batch[i][3] for i in xrange(batch_size)], np.int32)

    feed_dict = {
        placeholders['labels']: labels,
        placeholders['dropout']: dropout,
    }

    feed_dict.update(
        {placeholders['features'][i]: batch[i][0]
         for i in xrange(batch_size)})

    for j in xrange(levels):
        feed_dict.update({
            placeholders['adj_dist_{}'.format(j + 1)][i]: batch[i][1][j]
            for i in xrange(batch_size)
        })
        feed_dict.update({
            placeholders['adj_rad_{}'.format(j + 1)][i]: batch[i][2][j]
            for i in xrange(batch_size)
        })

    return feed_dict
