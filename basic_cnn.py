from __future__ import absolute_import, division, print_function

import tensorflow as tf 
import numpy as np 
import h5py
import generator as gen

tf.logging.set_verbosity(tf.logging.INFO)

FRAMES_TO_DISCARD = 20 #number of frames to desregard from the beginning and the end of the dataset
RUN_MODE = tf.estimator.ModeKeys.TRAIN
NO_OF_USERS_TRAIN = 10
NO_OF_USERS_EVAL = 3
NO_OF_TESTS = 10
TEST_DATA_PATH = "test/"
TRAIN_DATA_PATH = "/media/tkal976/Transcend/Tharindu/grey/Aeye_grey.h5"
LOGGING_ITERATIONS = 50
MINI_BATCH_SIZE = 128
NO_OF_STEPS = 20000

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
h_f_1 = 5
w_f_1 = 5
c_f_1 = 60
st_f_1 = 1
pd_f_1 = "SAME"
#conv2
n_f_2 = 64
h_f_2 = 5
w_f_2 = 5
c_f_2 = 64
st_f_2 = 1
pd_f_2 = "SAME"
#pool1
po_1 = 2
st_po_1 = 2
#conv3
n_f_3 = 128
h_f_3 = 3
w_f_3 = 3
c_f_2 = 64
st_f_3 = 1
pd_f_3 = "SAME"
#conv4
n_f_4 = 128
h_f_4 = 3
w_f_4 = 3
c_f_2 = 128
st_f_4 = 1
pd_f_4 = "SAME"
#pool2
po_2 = 2
st_po_2 = 2
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
po_h_3 = 3
po_w_3 = 2
st_po_h3 = 3
st_po_w3 = 2
# #conv8
# n_f_8 = 512
# h_f_8 = 3
# w_f_8 = 3
# st_f_8 = 1
# pd_f_8 = "SAME"
# #conv9
# n_f_9 = 512
# h_f_9 = 3
# w_f_9 = 3
# st_f_9 = 1
# pd_f_9 = "SAME"
# #conv10
# n_f_10 = 512
# h_f_10 = 3
# w_f_10 = 3
# st_f_10 = 1
# pd_f_10 = "SAME"
# #pool4
# po_4 = 2
# st_po_4 = 1
#dense1
de_1 = 8192
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

def load_file(datapath, mode = tf.estimator.ModeKeys.PREDICT):
	g = gen.generator(datapath)
	ds = tf.data.Dataset.from_generator(g, (tf.int8, tf.int8))
	value = ds.make_one_shot_iterator().get_next()
	data_sess = tf.Session()

	return data_sess, value

def file_handler(folderpath, mode = tf.estimator.ModeKeys.PREDICT):
	if(RUN_MODE == tf.estimator.ModeKeys.TRAIN):
		print("Trying to load training data file")
		datapath = folderpath
		#data_sess, value = load_file(datapath, mode)
		##run the neural netowrk in train mode
		run_nn(datapath)
	else:
		print("Trying to load test data file")
		datapath = folderpath
		#data_sess, value = load_file(datapath, mode)
		##run the neural netowrk in train mode
		run_nn(datapath)
	return 0

def Aeye_input_func_gen():
    shapes = ((300,440,60),(1,3))
    dataset = tf.data.Dataset.from_generator(generator=gen.generator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=shapes)
    dataset = dataset.batch(128)
    # dataset = dataset.repeat(20)
    iterator = dataset.make_one_shot_iterator()
    features_tensors, labels = iterator.get_next()
    print(labels.shape)
    features = {'x': features_tensors}
    return features, labels

def run_nn(datapath):
	# Create the Estimator
	# print("* * * * * trying to run session * * * * *")
	# try:
	# 	data = data_sess.run(value)
	# except tf.errors.OutOfRangeError:
	# 	print('done.')
	# print("* * * * * done - run session * * * * *")
	A_EYE_classifier = tf.estimator.Estimator(
		model_fn = cnn_model,
		model_dir = "model_dir/")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=LOGGING_ITERATIONS)
	# print("* * * * * trying normalize and read assign features * * * * *")
	# feature = tf.image.per_image_standardization(np.reshape(data[0], [-1, 300, 440, 60]))
	# print("* * * * * feature process done * * * * *")
	#train the model
	# train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
	# 	x = {"x": feature},
	# 	y = np.reshape(data[1], [-1, 1, 11]),#batch_size = MINI_BATCH_SIZE,
	# 	num_epochs = None,
	# 	shuffle = True)
	print("* * * * * infut function set up done* * * * *")
	A_EYE_classifier.train(
		input_fn = Aeye_input_func_gen,
		steps = NO_OF_STEPS,
		hooks = [logging_hook])

	print("* * * * * A_EYE_classifier.train done * * * * *")
	#Evatuate the model
	eval_input_fn = Aeye_input_func_gen()
	eval_results = A_EYE_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

def cnn_model(features, labels, mode = tf.estimator.ModeKeys.PREDICT):
	print("* * * * * setting input layer * * * * *")
	input_layer = features["x"]
	print("* * * * * setting input layer done* * * * *")
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
		pool_size = [po_h_3,po_w_3],
		strides = [st_po_h3,st_po_w3])

	flatten_layer = tf.reshape(pool3, [-1,25*55*256])

#Block 5
	dense1 = tf.layers.dense(
		inputs = flatten_layer,
		units = de_1,
		activation = tf.nn.relu)
	dropout1 = tf.layers.dropout(
		inputs = dense1,
		rate = dr_1,
		training = mode == tf.estimator.ModeKeys.TRAIN)
	dense2 = tf.layers.dense(
		inputs = dropout1,
		units = de_2,
		activation = tf.nn.relu)
	dropout2 = tf.layers.dropout(
		inputs = dense2,
		rate = dr_2,
		training = mode == tf.estimator.ModeKeys.TRAIN)

#Block 6_1
	logits1 = tf.layers.dense(
		inputs = dropout2,
		units = lo_1)

#Block 6_1
	logits2 = tf.layers.dense(
		inputs = dropout2,
		units = lo_2)

#Block 6_1
	logits3 = tf.layers.dense(
		inputs = dropout2,
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
	print(labels.shape)
	label_split = tf.reshape(labels,[-1,1,3])
	# print("label split *******************************************************")
	labels1 = label_split[0,0,0]
	labels2 = label_split[0,0,1]
	labels3 = label_split[0,0,2]

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
		file_handler(folderpath, mode = tf.estimator.ModeKeys.TRAIN)
		print("Train")
	else:
		folderpath = test_data_path
		file_handler(folderpath, mode = tf.estimator.ModeKeys.PREDICT)
		print("Test")

if __name__ == "__main__":
  tf.app.run()