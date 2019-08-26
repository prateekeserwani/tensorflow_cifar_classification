from dataloader import *
import tensorflow as tf
from model import CNN

data_root='./dataset/'
mode='train'
image_size=32
batch_size=5
num_classes=10	
training_iters=10000		
learning_rate = 0.1 	

def weight_and_biases():

	weights = {
	'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
	'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
	'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
	'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
	'out': tf.get_variable('W6', shape=(128,num_classes), initializer=tf.contrib.layers.xavier_initializer()), 
	}

	biases = {
	'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
	'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
	}
	return weights, biases


dl = dataloader(data_root,image_size,batch_size,mode, num_classes)

input_batch = tf.placeholder("float", [None, 32,32,3])
ground_truth = tf.placeholder("float", [None, num_classes])

weight, bias = weight_and_biases()
pred = CNN(input_batch, weight, bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ground_truth))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init) 
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
	summary_writer = tf.summary.FileWriter('./Output', sess.graph)
	for i in range(training_iters):
		image, gt = dl.create_batch()
		#print(gt)
		
		# Run optimization op (backprop).
		# Calculate batch loss and accuracy
		opt = sess.run(optimizer, feed_dict={input_batch: image, ground_truth: gt})

#		loss, acc = sess.run([cost, accuracy], feed_dict={x: image, y: gt})
		loss = sess.run([cost], feed_dict={input_batch: image, ground_truth: gt})

		print("Iter " + str(i))
		print(loss)
		print("Optimization Finished!")

		# Calculate accuracy for all 10000 mnist test images
#		test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,ground_truth : test_y})
		train_loss.append(loss)
#		test_loss.append(valid_loss)
#		train_accuracy.append(acc)
#		test_accuracy.append(test_acc)
#		print("Testing Accuracy:","{:.5f}".format(test_acc))
	summary_writer.close()
	
	

