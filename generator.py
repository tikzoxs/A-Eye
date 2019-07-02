import h5py
import tensorflow as tf
import numpy as np

class generator:
	def __init__(self, file):
		self.file = "/media/tkal976/Transcend/Tharindu/grey/Aeye_grey_wrong_labels.h5"
		# self.file = "/media/tkal976/Transcend/Tharindu/single_file/Aeye.h5"

	def __call__(self):
		with h5py.File(self.file, 'r') as h5f:
			X_dset = h5f['X']
			Y_dset = h5f['Y']
			print(Y_dset.shape)
			for i in range(X_dset.shape[0]):
				yield X_dset[i], np.argmax(Y_dset[i,:,0:5]), np.argmax(Y_dset[i,:,5:8]), np.argmax(Y_dset[i,:,8:11])