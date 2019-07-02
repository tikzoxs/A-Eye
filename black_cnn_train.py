from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import black_cnn as black_cnn
import black_cnn_eval as black_cnn_eval

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/Aeye_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def train_Aeye():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		with tf.device('/cpu:0'):
			features, scene_label, stress_label, focus_label = black_cnn.Aeye_train_input_func_gen()

		logits_scene, logits_stress, logits_focus = black_cnn.inference(features)

		loss = black_cnn.loss(logits_scene, logits_stress, logits_focus, scene_label, stress_label, focus_label)

		train_op = black_cnn.train(loss, global_step)

		total_parameters = 0
		for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
			shape = variable.get_shape()
			print(shape)
			print(len(shape))
			variable_parameters = 1
			for dim in shape:
				print(dim)
				variable_parameters *= dim.value
			print(variable_parameters)
			total_parameters += variable_parameters
		print("Total  trainable parameters : " + str(total_parameters / 1000000) + " Million")

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = float(duration / FLAGS.log_frequency)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
	                        'sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value,
	                               examples_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.train_dir,
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),tf.train.NanTensorHook(loss),_LoggerHook()],
			config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)
			save_path = saver.save(mon_sess, "/tmp/model.ckpt")
			print("Model saved in path: %s" % save_path)



def main(argv = None):
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train_Aeye()
	black_cnn_eval.eval_func()

if __name__ == '__main__':
	tf.app.run()