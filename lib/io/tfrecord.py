# record_iterator = tf.python_io.tf_record_iterator('test.tfrecords')
# print(record_iterator)
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#     features = example.features.feature['features'].bytes_list.value[0]
#     features = np.fromstring(features, dtype=np.float32)
#     features = np.reshape(features, (-1, 3))
#     print(features.shape)
# print(dist)
