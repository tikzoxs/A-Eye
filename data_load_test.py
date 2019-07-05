from __future__ import absolute_import, division, print_function

import tensorflow as tf 
import numpy as np 
import h5py
import generator as gen

datapath = "/home/tkal976/Desktop/grey/Aeye_grey.h5"
g = gen.generator(datapath)
ds = tf.data.Dataset.from_generator(g, (tf.int8, tf.int8, tf.int8, tf.int8))

value = ds.make_one_shot_iterator().get_next()

sess = tf.Session()

# Example on how to read elements
while True:
    try:
        data = sess.run(value)
        print(data[0].shape, data[1], data[2], data[3])
    except tf.errors.OutOfRangeError:
        print('done.')
        break