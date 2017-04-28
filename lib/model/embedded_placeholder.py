from six.moves import xrange

import tensorflow as tf


def embedded_placeholders(batch_size, levels, num_features, num_labels):
    placeholders = {
        'features': [
            tf.placeholder(tf.float32, [None, num_features],
                           'features_{}'.format(i + 1))
            for i in xrange(batch_size)
        ],
        'labels':
        tf.placeholder(tf.int32, [batch_size, 10], 'labels'),
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


def embedded_feed_dict(placeholders, features, labels, adjs_dist, adjs_rad,
              dropout=0.0):
    feed_dict = {
        placeholders['labels']: labels,
        placeholders['dropout']: dropout,
    }

    feed_dict.update({
        placeholders['features'][i]: features[i]
        for i in xrange(len(features))
    })

    for j in xrange(len(adjs_dist[0]) - 1):
        feed_dict.update({
            placeholders['adj_dist_{}'.format(j + 1)][i]: adjs_dist[i][j]
            for i in xrange(len(features))
        })
        feed_dict.update({
            placeholders['adj_rad_{}'.format(j + 1)][i]: adjs_rad[i][j]
            for i in xrange(len(features))
        })

    return feed_dict
