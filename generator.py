import h5py
import tensorflow as tf
import numpy as np

class generator:
	def __init__(self, file):
		self.file = "/home/tkal976/Desktop/grey/Aeye_grey.h5"

	def __call__(self):
		with h5py.File(self.file, 'r') as h5f:
			X_dset = h5f['X']
			Y_dset = h5f['Y']
			print(X_dset.shape[0])
			for j in range(100):
				for i in range(X_dset.shape[0] - X_dset.shape[0] % 64):
					yield X_dset[i], np.argmax(Y_dset[i,:,0:5]), np.argmax(Y_dset[i,:,5:8]), np.argmax(Y_dset[i,:,8:11])
