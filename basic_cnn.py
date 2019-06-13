from __future__ import absolute_import, division, print_function

import tensorflow as tf 
import numpy as np 
import h5py

tf.logging.set_verbosity(tf.logging.INFO)

FRAMES_TO_DISCARD = 20 #number of frames to desregard from the beginning and the end of the dataset
RUN_MODE = tf.estimator.ModeKeys.PREDICT
NO_OF_USERS = 10
NO_OF_TESTS = 10
TEST_DATA_PATH = "test/"
TRAIN_DATA_PATH = "train/"

#### HYPERPARAMETERS ####
INPUT_CONCATANATION = 60 #number of Frames considered for a decision
object_loss_coefficient = 0.375
stress_loss_coefficient = 0.375
focus_loss_coefficient = 1 - object_loss_coefficient - stress_loss_coefficient
learning_rate = 0.001
# n_f = number of filters
# h_f = height of filter
# w_f = width of filter
# st_f = stride of filter
# pd_f = padding of filter
# po_ = pooling kernel size
# st_po = stride of pooling
#conv1
n_f_1 = 64
h_f_1 = 7
w_f_1 = 7
st_f_1 = 1
pd_f_1 = "SAME"
#conv2
n_f_2 = 64
h_f_2 = 7
w_f_2 = 7
st_f_2 = 1
pd_f_2 = "SAME"
#pool1
po_1 = 2
st_po_1 = 1
#conv3
n_f_3 = 128
h_f_3 = 5
w_f_3 = 5
st_f_3 = 1
pd_f_3 = "SAME"
#conv4
n_f_4 = 128
h_f_4 = 5
w_f_4 = 5
st_f_4 = 1
pd_f_4 = "SAME"
#pool2
po_2 = 2
st_po_2 = 1
#conv5
n_f_5 = 256
h_f_5 = 5
w_f_5 = 5
st_f_5 = 1
pd_f_5 = "SAME"
#conv6
n_f_6 = 256
h_f_6 = 5
w_f_6 = 5
st_f_6 = 1
pd_f_6 = "SAME"
#conv7
n_f_7 = 256
h_f_7 = 5
w_f_7 = 5
st_f_7 = 1
pd_f_7 = "SAME"
#pool3
po_3 = 2
st_po_3 = 1
#conv8
n_f_8 = 512
h_f_8 = 3
w_f_8 = 3
st_f_8 = 1
pd_f_8 = "SAME"
#conv9
n_f_9 = 512
h_f_9 = 3
w_f_9 = 3
st_f_9 = 1
pd_f_9 = "SAME"
#conv10
n_f_10 = 512
h_f_10 = 3
w_f_10 = 3
st_f_10 = 1
pd_f_10 = "SAME"
#pool4
po_4 = 2
st_po_4 = 1
#dense1
de_1 = 4096
dr_1 = 0.4
#dense2
de_2 = 1024
dr_2 = 0.4
#logits 1
lo_1 = 5
#logits 2
lo_2 = 3
#logits 3
lo_3 = 3

def create_input(X, Y, mode = tf.estimator.ModeKeys.PREDICT):
	n_x = X.shape[0] #no of images
	h_x = X.shape[1] #height of image
	w_x = X.shape[2] #width of image
	n_c = X.shape[3] #number of channels in an image

	assert n_x == Y.shape[0], "Data and label size mismatch !!!"

	excess = (n_x - 2 * FRAMES_TO_DISCARD) % INPUT_CONCATANATION
	inputs = X[FRAMES_TO_DISCARD + excess:-FRAMES_TO_DISCARD,:,:,:].flatten()
	input_array = np.reshape(inputs,(inputs.shape[0]/INPUT_CONCATANATION, INPUT_CONCATANATION * h_x * w_x * n_c))
	labels = Y[0:inputs.shape[0]/INPUT_CONCATANATION,:,:]

	if(mode == tf.estimator.ModeKeys.TRAIN):
		return input_array, labels
	return input_array

def load_file(datapath):
	try:
		with h5py.File(datapath, mode='r') as h5f:
				X = h5f['X']
				Y = h5f['Y']
		return X, Y
	except:
		print("No such file")

def file_handler(folderpath, mode = tf.estimator.ModeKeys.PREDICT):
	if(RUN_MODE == tf.estimator.ModeKeys.TRAIN):
		for i in range(1,NO_OF_USERS+1):
			for j in range(NO_OF_TESTS):
				datapath = folderpath + str(i) + "_" + str(j) + ".h5"
				X, Y = load_file(datapath)
				input_array, labels = create_input(X, Y, mode = tf.estimator.ModeKeys.TRAIN)
				##run the neural netowrk in train mode
	else:
		datapath = folderpath + "test_data.h5"
		X, Y = load_file(datapath)
		input_array = create_input(X, Y, mode = tf.estimator.ModeKeys.PREDICT)
		##run the neural netowrk in test mode
	return 0

def model(X, Y, mode = mode = tf.estimator.ModeKeys.PREDICT):

	input_layer = X

#Block 1
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = n_f_1,
		kernel_size = [h_f_1, w_f_1],
		strides = st_f_1,
		padding = pd_f_1,
		activation = tf.nn.relu)
	conv2 = tf.layers.conv2d(
		inputs = conv1,
		filters = n_f_2,
		kernel_size = [h_f_2, w_f_2],
		strides = st_f_2,
		padding = pd_f_2,
		activation = tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = po_1,
		strides = st_po_1)

#Block 2
	conv3 = tf.layers.conv2d(
		inputs = pool1,
		filters = n_f_3,
		kernel_size = [h_f_3, w_f_3],
		strides = st_f_3,
		padding = pd_f_3,
		activation = tf.nn.relu)
	conv4 = tf.layers.conv2d(
		inputs = conv3,
		filters = n_f_4,
		kernel_size = [h_f_4, w_f_4],
		strides = st_f_4,
		padding = pd_f_4,
		activation = tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(
		inputs = conv4,
		pool_size = po_2,
		strides = st_po_2)

#Block 3
	conv5 = tf.layers.conv2d(
		inputs = pool2,
		filters = n_f_5,
		kernel_size = [h_f_5, w_f_5],
		strides = st_f_5,
		padding = pd_f_5,
		activation = tf.nn.relu)
	conv6 = tf.layers.conv2d(
		inputs = conv5,
		filters = n_f_6,
		kernel_size = [h_f_6, w_f_6],
		strides = st_f_6,
		padding = pd_f_6,
		activation = tf.nn.relu)
	conv7 = tf.layers.conv2d(
		inputs = conv6,
		filters = n_f_7,
		kernel_size = [h_f_7, w_f_7],
		strides = st_f_7,
		padding = pd_f_7,
		activation = tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(
		inputs = conv7,
		pool_size = po_3,
		strides = st_po_3)

#Block 4
	conv8 = tf.layers.conv2d(
		inputs = pool3,
		filters = n_f_8,
		kernel_size = [h_f_8, w_f_8],
		strides = st_f_8,
		padding = pd_f_8,
		activation = tf.nn.relu)
	conv9 = tf.layers.conv2d(
		inputs = conv8,
		filters = n_f_9,
		kernel_size = [h_f_9, w_f_9],
		strides = st_f_9,
		padding = pd_f_9,
		activation = tf.nn.relu)
	conv10 = tf.layers.conv2d(
		inputs = conv9,
		filters = n_f_10,
		kernel_size = [h_f_10, w_f_10],
		strides = st_f_10,
		padding = pd_f_10,
		activation = tf.nn.relu)
	pool4 = tf.layers.max_pooling2d(
		inputs = conv10,
		pool_size = po_4,
		strides = st_po_4)
	flatten_layer = tf.reshape(pool4, [1,-1])

#Block 5
	dense1 = tf.layers.dense(
		inpus = flatten_layer,
		units = de_1,
		activation = tf.nn.relu)
	dropout1 = tf.layers.dropout(
		inputs = dense1,
		rate = dr_1,
		training = mode == tf.estimator.ModeKeys.TRAIN)
	dense2 = tf.layers.dense(
		inpus = dropout1,
		units = de_2,
		activation = tf.nn.relu)
	dropout2 = tf.layers.dropout(
		inputs = dense2,
		rate = dr_2,
		training = mode == tf.estimator.ModeKeys.TRAIN)

#Block 6_1
	logits1 = tf.layers.dense(
		inpus = dropout2,
		units = lo_1)

#Block 6_1
	logits2 = tf.layers.dense(
		inpus = dropout2,
		units = lo_2)

#Block 6_1
	logits2 = tf.layers.dense(
		inpus = dropout2,
		units = lo_3)

#Predictions in PREDICT mode
	predictions1 = {
			"classes" : tf.argmax(input = logits1, axis = 1),
			"probabilities" : tf.nn.softmax(logits1, name = "softmax_tensor1")
		}

	predictions2 = {
			"classes" : tf.argmax(input = logits2, axis = 1),
			"probabilities" : tf.nn.softmax(logits2, name = "softmax_tensor2")
		}
	predictions3 = {
			"classes" : tf.argmax(input = logits3, axis = 1),
			"probabilities" : tf.nn.softmax(logits3, name = "softmax_tensor3")
		}

	if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions1), 
			tf.estimator.EstimatorSpec(mode = mode, predictions = predictions2), 
			tf.estimator.EstimatorSpec(mode = mode, predictions = predictions3)

#labels 
	labels1 = Y[0:5]
	labels2 = Y[5:8]
	labels3 = Y[8:11]

#Calculate loss
	loss1 = tf.losses.sparse_softmax_cross_entropy(labels = labels1, logits = logits1)
	loss2 = tf.losses.sparse_softmax_cross_entropy(labels = labels2, logits = logits2)
	loss3 = tf.losses.sparse_softmax_cross_entropy(labels = labels3, logits = logits3)
	loss = object_loss_coefficient * loss1 + stress_loss_coefficient * loss2 + focus_loss_coefficient * loss3

#when training
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		train_op = optimizer.minimize(
			loss = loss,
			global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)


#when evaluating 
	eval_metric_ops = {
		"accuracy" : tf.metrics.accuracy(
			labels = labels,
			predictions = predictions["classes"])
	}
	return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(args):
	test_data_path = TEST_DATA_PATH
	train_data_path = TRAIN_DATA_PATH
	if(RUN_MODE == tf.estimator.ModeKeys.TRAIN):
		folderpath = train_data_path
		file_handler(folderpath, mode = tf.estimator.ModeKeys.TRAIN):
	else:
		folderpath = test_data_path
		file_handler(folderpath, mode = tf.estimator.ModeKeys.PREDICT):
