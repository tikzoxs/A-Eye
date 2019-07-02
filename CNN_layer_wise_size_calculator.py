from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import numpy as np 
import tensorflow as tf 
import generator as geny
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#uncomment for GPU 
#tf.test.gpu_device_name()
FLAGS = tf.app.flags.FLAGS

RUN_MODE = tf.estimator.ModeKeys.TRAIN
TEST_DATA_PATH = "test/"
TRAIN_DATA_PATH = "/home/tkal976/Desktop/grey/Aeye_grey.h5"

#### BASIC PARAMETERS ####
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('logging_iterations', 50,
                            """Number of iterations to wait till logging.""")
tf.app.flags.DEFINE_integer('no_epochs', 100,
                            """Number of epochs to train over datset.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('image_width', 440,
                            """image width.""")
tf.app.flags.DEFINE_integer('image_height', 300,
                            """image height.""")
tf.app.flags.DEFINE_integer('image_channels', 60,
                            """number of channels.""")
tf.app.flags.DEFINE_integer('label_rows', 1,
                            """dimension 1 of a single label array.""")
tf.app.flags.DEFINE_integer('label_cols', 3,
                            """dimension 2 of a single label array.""")

TOWER_NAME = 'tower'

#### HYPERPARAMETERS ####
NO_SCENES = 5
NO_STRESS = 3
NO_FOUCS = 3
INPUT_CONCATANATION = 60 #number of Frames considered for a decision
SCENE_LOSS_COEFFICIENT = 0.375
STRESS_LOSS_COEFFICIENT = 0.375
FOCUS_LOSS_COEFFICIENT = 1 - SCENE_LOSS_COEFFICIENT - STRESS_LOSS_COEFFICIENT
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2 ** 10
###############################
## n_f = number of filters ####
## h_f = height of filter #####
## w_f = width of filter ######
## st_f = stride of filter ####
## pd_f = padding of filter3 ##
## po_ = pooling kernel size ##
## st_po = stride of pooling ##
###############################
#conv1
n_f_1 = 64
h_f_1 = 5
w_f_1 = 5
c_f_1 = 60
st_f_1 = [1,1,1,1]
pd_f_1 = "SAME"
#conv2
n_f_2 = 64
h_f_2 = 5
w_f_2 = 5
c_f_2 = 64
st_f_2 = [1,1,1,1]
pd_f_2 = "SAME"
#pool1
po_h_1 = 3
po_w_1 = 3
st_po_h_1 = 2
st_po_w_1 = 2
pd_po_1 = 'SAME'
#conv3
n_f_3 = 128
h_f_3 = 3
w_f_3 = 3
c_f_3 = 64
st_f_3 = [1,1,1,1]
pd_f_3 = "SAME"
#conv4
n_f_4 = 128
h_f_4 = 3
w_f_4 = 3
c_f_4 = 128
st_f_4 = [1,1,1,1]
pd_f_4 = "SAME"
#pool2
po_h_2 = 3
po_w_2 = 3
st_po_h_2 = 3
st_po_w_2 = 3
pd_po_2 = 'SAME'
#conv5
# n_f_5 = 256
# h_f_5 = 3
# w_f_5 = 3
# c_f_5 = 128
# st_f_5 = [1,1,1,1]
# pd_f_5 = "SAME"
#conv6
n_f_6 = 256
h_f_6 = 3
w_f_6 = 3
c_f_6 = 128
st_f_6 = [1,2,2,1]
pd_f_6 = "SAME"
#conv7
n_f_7 = 512
h_f_7 = 3
w_f_7 = 3
c_f_7 = 256
st_f_7 = [1,2,2,1]
pd_f_7 = "SAME"
#conv8
n_f_8 = 512
h_f_8 = 3
w_f_8 = 3
c_f_8 = 256
st_f_8 = [1,2,2,1]
pd_f_8 = "SAME"
#pool3
po_h_3 = 3
po_w_3 = 3
st_po_h_3 = 3
st_po_w_3 = 3
pd_po_3 = 'SAME'
#dense1
de_1 = 2048
dr_1 = 0.4
#dense2
de_2 = 1024
dr_2 = 0.4
#softmax
de_scene = 5
de_stress = 3
de_focus = 3
#logits 1
lo_1 = 5
#logits 2
lo_2 = 3
#logits 3
lo_3 = 3

def _activation_summary(x):
	"""Helper to create summaries for activations.
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.
	Args:
	x: Tensor
	Returns:
	nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _create_cpu_variable(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		if initializer is not None and not callable(initializer):
			var = tf.get_variable(name, initializer=initializer, dtype=dtype)
		else:
			var = tf.get_variable(name, shape = shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay_option(name, shape, stddev, wieght_decay_parameter):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	truncated_normal = tf.truncated_normal_initializer()#(stddev = stddev)
	print("******************************************************************")
	print(shape)
	var = _create_cpu_variable(name, shape, truncated_normal(shape, dtype = dtype))
	if wieght_decay_parameter is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wieght_decay_parameter, name = 'weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv_2d(layer_input, width, height, channels, filters, strides, padding, layer_name, stddev, wieght_decay_parameter):
	with tf.variable_scope(layer_name) as scope:
		kernel = _variable_with_weight_decay_option(
			'weights',
			shape = [width, height, channels, filters],
			stddev = stddev,
			wieght_decay_parameter = wieght_decay_parameter)
		conv_out = tf.nn.conv2d(layer_input, kernel, strides, padding)
		biases = _create_cpu_variable('biases', [filters], tf.constant_initializer(0.1))
		biases_added = tf.nn.bias_add(conv_out, biases)
		layer_out = tf.nn.relu(biases_added, name = scope.name)
		_activation_summary(layer_out)
	return layer_out

def max_pool(layer_input, pool_h, pool_w, stride_h, stride_w, padding, name, avg_batches = 1, avg_channels = 1, strie_of_batch = 1, stride_of_channels = 1):
	return tf.nn.max_pool(
		layer_input, 
		ksize = [avg_batches, pool_h, pool_w, avg_channels],
		strides = [strie_of_batch, stride_h, stride_w, stride_of_channels],
		padding = padding,
		name = name)

def normalize_layer(layer_input, depth_radius = 5, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75, name = 'norm'):
	return tf.nn.lrn(layer_input, depth_radius = depth_radius, bias = bias, alpha = alpha / 9.0, beta = beta, name = name)

def flatten_layer_followed_by_dense(layer_input, length_this_layer, layer_name, stddev = 0.04, wieght_decay_parameter = 0.004, initializer_parameter = 0.1):
	with tf.variable_scope(layer_name) as scope:
		flatten = tf.keras.layers.Flatten()(layer_input)
		length_prev_layer = flatten.get_shape()[1].value
	return length_prev_layer

def dense_layer(layer_input, length_prev_layer, length_this_layer, layer_name, is_output_layer = False, stddev = 0.04, wieght_decay_parameter = 0.004, initializer_parameter = 0.1):
	with tf.variable_scope(layer_name) as scope:
		this_layer = _variable_with_weight_decay_option(
			'weights',
			shape = [length_prev_layer, length_this_layer],
			stddev =stddev,
			wieght_decay_parameter = wieght_decay_parameter)
		biases = _create_cpu_variable(
			'biases',
			[length_this_layer],
			tf.constant_initializer(initializer_parameter))
		dense_mul = tf.add(tf.matmul(layer_input, this_layer), biases)
		if(is_output_layer):
			_activation_summary(dense_mul)
		else:
			dense_out = tf.nn.relu(dense_mul, name = scope.name)
			_activation_summary(dense_out)
	if(is_output_layer):
		return dense_mul
	else:
		return dense_out

def inference(features):
	# Block 1
	conv_l_1 = conv_2d(features["x"], w_f_1, h_f_1, c_f_1, n_f_1, st_f_1, pd_f_1, 'conv_l_1', 5e-2, None)
	norm_l_1 = normalize_layer(conv_l_1, name = 'norm_l_1')
	conv_l_2 = conv_2d(norm_l_1, w_f_2, h_f_2, c_f_2, n_f_2, st_f_2, pd_f_2, 'conv_l_2', 5e-2, None)
	norm_l_2 = normalize_layer(conv_l_2, name = 'norm_l_2')
	pool_l_1 = max_pool(norm_l_2, po_h_1, po_w_1, st_po_h_1, st_po_w_1, pd_po_1, 'pool_l_1')

	# Block 2
	conv_l_3 = conv_2d(pool_l_1, w_f_3, h_f_3, c_f_3, n_f_3, st_f_3, pd_f_3, 'conv_l_3', 5e-2, None)
	norm_l_3 = normalize_layer(conv_l_3, name = 'norm_l_3')
	conv_l_4 = conv_2d(norm_l_3, w_f_4, h_f_4, c_f_4, n_f_4, st_f_4, pd_f_4, 'conv_l_4', 5e-2, None)
	norm_l_4 = normalize_layer(conv_l_4, name = 'norm_l_4')
	# conv_l_5 = conv_2d(norm_l_4, w_f_5, h_f_5, c_f_5, n_f_5, st_f_5, pd_f_5, 'conv_l_5', 5e-2, None)
	# norm_l_5 = normalize_layer(conv_l_5, name = 'norm_l_5')
	pool_l_2 = max_pool(norm_l_4, po_h_2, po_w_2, st_po_h_2, st_po_w_2, pd_po_2, 'pool_l_2')

	# Block 3
	conv_l_6 = conv_2d(pool_l_2, w_f_6, h_f_6, c_f_6, n_f_6, st_f_6, pd_f_6, 'conv_l_6', 5e-2, None)
	norm_l_6 = normalize_layer(conv_l_6, name = 'norm_l_6')
	conv_l_7 = conv_2d(norm_l_6, w_f_7, h_f_7, c_f_7, n_f_7, st_f_7, pd_f_7, 'conv_l_7', 5e-2, None)
	norm_l_7 = normalize_layer(conv_l_7, name = 'norm_l_7')
	conv_l_8 = conv_2d(norm_l_7, w_f_8, h_f_8, c_f_8, n_f_8, st_f_8, pd_f_8, 'conv_l_8', 5e-2, None)
	norm_l_8 = normalize_layer(conv_l_8, name = 'norm_l_8')
	pool_l_3 = max_pool(norm_l_8, po_h_3, po_w_3, st_po_h_3, st_po_w_3, pd_po_3, 'pool_l_2')

	dlength = flatten_layer_followed_by_dense(pool_l_3, de_1, 'dense_l_1')
	# Block 4

	print(dlength)


test_array = tf.cast(np.random.randn(1,300,440,60),tf.float32)
features = {"x": test_array}
inference(features)

