# -*- coding: UTF-8 -*-

import tensorflow as tf
import data_processor
import os
import numpy as np


embedding_size = 100
batch_size = 50
sequence_length = 200
filter_sizes=[1,2,3,5]
num_filters = 500
loss_margin = 0.05
learning_rate = 0.001
num_epoch = 100
eval_every = 200
ratio = batch_size
test_size = 100


# Build vocabulary first
print("------ cnn.py~~ Loading data ------")
filePath = '/home/sherrie/PycharmProjects/cnnDemo_core/'
vocab = data_processor.buildVocab(filePath)
vocab_size = len(vocab)


graph = tf.Graph()
with graph.as_default():
	# define the parameters of cnn
	w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
	                name="W")  # W is embedding matrixd
	w_conv = list()
	b_conv = list()
	for filter_size in filter_sizes:
		filter_shape = [filter_size, embedding_size, 1, num_filters]
		w_conv.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1)))
		b_conv.append(tf.Variable(tf.constant(0.1, shape=[num_filters])))

	# Placeholders for varialbes. input_x1,input_x2,input_x3, which are respectively question, positive answer, negative answer.
	input_x1 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x1")
	input_x2 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x2")
	input_x3 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x3")

	# embedding layer, embed the words into vectors and expand them to 4 dim tensors for cnn architecture.
	with tf.device('/cpu:0'), tf.name_scope("embedding"):
		embedded_chars1 = tf.nn.embedding_lookup(w_embedding, input_x1)
		embedded_chars2 = tf.nn.embedding_lookup(w_embedding, input_x2)
		embedded_chars3 = tf.nn.embedding_lookup(w_embedding, input_x3)
		embedded_chars1_expanded = tf.expand_dims(embedded_chars1,
		                                               -1)  # add a dim at the end of the variable. input of image conv has 4 dims.
		embedded_chars2_expanded = tf.expand_dims(embedded_chars2, -1)
		embedded_chars3_expanded = tf.expand_dims(embedded_chars3, -1)
	pooled_outputs1 = []
	pooled_outputs2 = []
	pooled_outputs3 = []
	for i in range((len(filter_sizes))):
		conv = tf.nn.conv2d(embedded_chars1_expanded, w_conv[i], strides=[1, 1, 1, 1],
		                    padding="VALID", name="conv")
		h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
		pooled = tf.nn.max_pool(
			h,
			ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID',
			name="pool")  # shape of pooled is [batch_size,1,1,num_filters]
		pooled_outputs1.append(pooled)
		conv = tf.nn.conv2d(embedded_chars2_expanded,  w_conv[i], strides=[1, 1, 1, 1],
		                    padding="VALID", name="conv")
		h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
		pooled = tf.nn.max_pool(
			h,
			ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID',
			name="pool")
		pooled_outputs2.append(pooled)

		conv = tf.nn.conv2d(embedded_chars3_expanded,  w_conv[i], strides=[1, 1, 1, 1],
		                    padding="VALID", name="conv")
		# Apply nonlinearity
		h = tf.nn.relu(tf.nn.bias_add(conv, b_conv[i]), name="relu")
		# Max-pooling over the outputs
		pooled = tf.nn.max_pool(
			h,
			ksize=[1, sequence_length - filter_sizes[i] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID',
			name="pool")
		pooled_outputs3.append(pooled)

	# reshape the outputs to combine all the pooled features
	num_filters_total = num_filters * len(filter_sizes)
	h_pooled1 = tf.concat(pooled_outputs1 ,3)  # the 4th dim corresponds to num_filters
	h_pooled1_flat = tf.reshape(h_pooled1, [-1, num_filters_total])
	h_pooled2 = tf.concat(pooled_outputs2, 3)
	h_pooled2_flat = tf.reshape(h_pooled2, [-1, num_filters_total])
	h_pooled3 = tf.concat(pooled_outputs3, 3)
	h_pooled3_flat = tf.reshape(h_pooled3, [-1, num_filters_total])

	len_pooled1 = tf.sqrt(
		tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled1_flat), 1))  # length of quesiton vectors
	len_pooled2 = tf.sqrt(
		tf.reduce_sum(tf.multiply(h_pooled2_flat, h_pooled2_flat), 1))  # length of positive answer vectors
	len_pooled3 = tf.sqrt(
		tf.reduce_sum(tf.multiply(h_pooled3_flat, h_pooled3_flat), 1))  # length of negative answer vectors
	mul_12 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled2_flat),
	                       1)  # wisely multiple vectors
	mul_13 = tf.reduce_sum(tf.multiply(h_pooled1_flat, h_pooled3_flat), 1)

	# output
	with tf.name_scope("output"):
		cos_12 = tf.div(mul_12, tf.multiply(len_pooled1, len_pooled2),
		                     name="scores")
		cos_13 = tf.div(mul_13, tf.multiply(len_pooled1, len_pooled3), )

	zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
	margin = tf.constant(loss_margin, shape=[batch_size], dtype=tf.float32)

	with tf.name_scope("loss"):
		losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(cos_12, cos_13)))
		loss = tf.reduce_sum(losses)
		loss = tf.div(loss, batch_size)

	# accuracy
	with tf.name_scope("accuracy"):
		correct = tf.equal(zero, losses)
		accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")


	# 显示/保存测试数据
	def save_test_data(y1, y2, y3, i):
		sen_y1 = data_processor.getSentence(y1, vocab)[0]
		sen_y2 = data_processor.getSentence(y2, vocab)[0]
		sen_y3 = data_processor.getSentence(y3, vocab)
		data_processor.saveData('\nQuestion ' + str(i + 1) + ':\n' + sen_y1)
		data_processor.saveData('\nPositive Answer:\n' + sen_y2)
		data_processor.saveData('\nNegative Answers:')
		for j in range(4):
			data_processor.saveData('\n' + str(j + 1) + ' ' + sen_y3[j])
		return


	# define a validation/test step
	def test_step(input_y1, input_y2, input_y3, flag_list, sess):
		feed_dict = dict()
		feed_dict = {
			input_x1: input_y1,
			input_x2: input_y2,
			input_x3: input_y3}

		correct_flag = 0
		test_losses_ = sess.run(losses, feed_dict)
		test_losses_ = test_losses_*flag_list
		test_loss_ = sum(test_losses_)    # add all the losses
		if test_loss_ == 0.0:
			correct_flag = 1
		cos_pos_, cos_neg_, accuracy_ = sess.run([cos_12, cos_13, accuracy], feed_dict)
		data_processor.saveFeatures(cos_pos_, cos_neg_, test_loss_, accuracy_)
		return correct_flag


	def test():
		correct_num = int(0)
		for i in range(test_size):
			batch_y1, batch_y2, batch_y3, flag_list = data_processor.loadValData(vocab, filePath, sequence_length, ratio)      # batch_size*seq_len
			# 显示/保存测试数据
			save_test_data(batch_y1, batch_y2, batch_y3,i)
			correct_flag = test_step(batch_y1, batch_y2, batch_y3, flag_list, sess)
			correct_num += correct_flag
		print ('correct_num',correct_num)
		acc = correct_num / float(test_size)
		return acc


	with tf.Session() as sess:
		# Before we can train our model we also need to initialize the variables in our graph.
		global_step = tf.Variable(0, name="global_step",
		                          trainable=False)  # The global step will be automatically incremented by one every time you execute　a train loop
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		grads_and_vars = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Output directory for models and summaries
		checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints_train"))
		print("Writing to {}\n".format(checkpoint_dir))
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables())
		sess.run(tf.global_variables_initializer())

		feed_dict = dict()
		batch_x1, batch_x2, batch_x3 = data_processor.loadTrainData(vocab, filePath, sequence_length,
		                                                            batch_size, 0)  # batch_size*seq_len
		feed_dict = {
			input_x1: batch_x1,
			input_x2: batch_x2,
			input_x3: batch_x3}
		losses_ = sess.run(losses, feed_dict)
		step_, cos_pos_, cos_neg_, loss_, accuracy_ = sess.run(
			[global_step, cos_12, cos_13, loss, accuracy], feed_dict)
		print ('=' * 10 + 'step{}, loss = {}, acc={}'.format(step_, loss_, accuracy_))  # loss for all batches
		while step_ < num_epoch*20000/batch_size:
			feed_dict=dict()
			batch_x1, batch_x2, batch_x3 = data_processor.loadTrainData(vocab, filePath, sequence_length,
			                                                            batch_size, step_)  # batch_size*seq_len

			feed_dict = {
				input_x1: batch_x1,
				input_x2: batch_x2,
				input_x3: batch_x3}
			losses_ = sess.run(losses, feed_dict)
			_, step_, cos_pos_, cos_neg_, loss_, accuracy_ = sess.run(
				[train_op, global_step, cos_12, cos_13, loss, accuracy], feed_dict)
			print ('=' * 10 + 'step{}, loss = {}, acc={}'.format(step_,loss_, accuracy_))        # loss for all batches
			if step_%eval_every==0:
				print('\n============================> begin to test ')
				acc_ = test()
				# acc = test_for_bilstm.test()
				print(
				'--------The test result among the test data sets: acc = {}, test size = {}, test ratio = {}----------'.format(
					acc_, test_size, ratio))
				path = saver.save(sess, checkpoint_dir + '/step' + str(step_) +'_loss'+ str(loss_) +'_trainAcc'+ str(accuracy_)+ '_testAcc' + str(acc_))
				print("Save checkpoint(model) to {}".format(path))


