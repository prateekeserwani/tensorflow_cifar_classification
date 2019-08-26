import tensorflow as tf


def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x) 

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def CNN(x, weights, biases):
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)

	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3, k=2)


	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)

	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out




