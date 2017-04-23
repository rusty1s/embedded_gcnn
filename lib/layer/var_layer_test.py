# def test_bias_constant(self):
#     lap = tf.constant(1)

#     layer1 = ChebyshevGCNN(2, 3, lap, degree=3, name='bias_1')
#     layer2 = ChebyshevGCNN(
#         2, 3, lap, degree=3, bias_constant=1.0, name='bias_2')

#     with self.test_session() as sess:
#         sess.run(tf.global_variables_initializer())

#         self.assertAllEqual(layer1.vars['bias'].eval(),
#                             np.array([0.1, 0.1, 0.1], dtype=np.float32))
#         self.assertAllEqual(layer2.vars['bias'].eval(),
#                             np.array([1.0, 1.0, 1.0], dtype=np.float32))
