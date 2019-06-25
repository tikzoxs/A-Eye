import numpy as np 
import tensorflow as tf 
import generator as geny

#uncomment for GPU 
#tf.test.gpu_device_name()
FLAGS = tf.app.flags.FLAGS

RUN_MODE = tf.estimator.ModeKeys.TRAIN
TEST_DATA_PATH = "test/"
TRAIN_DATA_PATH = "/media/tkal976/Transcend/Tharindu/grey/Aeye_grey.h5"

#### BASIC PARAMETERS ####
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('logging_iterations', 50,
                            """Number of iterations to wait till logging.""")
tf.app.flags.DEFINE_integer('no_epochs', 2000,
                            """Number of epochs to train over datset.""")
tf.app.flags.DEFINE_boolean('use_fp16', True,
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


#### HYPERPARAMETERS ####
INPUT_CONCATANATION = 60 #number of Frames considered for a decision
object_loss_coefficient = 0.375
stress_loss_coefficient = 0.375
focus_loss_coefficient = 1 - object_loss_coefficient - stress_loss_coefficient
learning_rate = 0.001
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
    	var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  	return var

def variable_with_weight_decay_option(name, shpae, stddev, wieght_decay_parameter):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _create_cpu_variable(name, shape, tf.truncated_normal_initializer(stddev = stddev, dtype = dtype))
	if wieght_decay_parameter is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wieght_decay_parameter, name = 'weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv_2d(layer_input, width, height, channels, filters, strides, padding, stddev):
	with tf.variable_scope(layer_name) as scope:
		kernel = variable_with_weight_decay_option(
			'weights',
			shape = [width, height, channels, filters],
			stddev = stddev,
			wieght_decay_parameter = wieght_decay_parameter)
		conv_out = tf.nn.conv2d(layer_input, kernel, strides, padding)
		biases = _create_cpu_variable('biases', [filters], tf.constant_initializer(0.0))
		biases_added = tf.nn.bias_add(conv_out, biases)
		layer_out = tf.nn.relu(biases_added, name = scope.name)

def max_pool(layer_input, pool_h, pool_w, stride_h, stride_w, padding, name, avg_batches = 1, avg_channels = 1, strie_of_batch = 1, stride_of_channels = 1):
	return tf.nn.max_pool(
		layer_input, 
		ksize = [avg_batches, pool_h, pool_w, avg_channels],
		strides = [strie_of_batch, stride_h, stride_w, stride_of_channels],
		padding = padding,
		name = name)

def normalize_layer(layer_input, depth_radius = 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1'):
	return tf.nn.lrn(pool1, depth_radius = depth_radius, bias = bias, alpha = alpha / 9.0, beta = beta, name = name)

def Aeye_train_input_func_gen():
    shapes = ((image_height, image_width, image_channels),(label_rows, label_cols))
    dataset = tf.data.Dataset.from_generator(generator=gen.generator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=shapes)
    dataset = dataset.batch(128)
    iterator = dataset.make_one_shot_iterator()
    features_tensors, labels = iterator.get_next()
    print(labels.shape)
    features = {'x': features_tensors}
    return features, labels

def Aeye_eval_input_func_gen():
    shapes = ((image_height, image_width, image_channels),(label_rows, label_cols))
    dataset = tf.data.Dataset.from_generator(generator=gen.generator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=shapes)
    dataset = dataset.batch(128)
    iterator = dataset.make_one_shot_iterator()
    features_tensors, labels = iterator.get_next()
    print(labels.shape)
    features = {'x': features_tensors}
    return features, labels