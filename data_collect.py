   # X contains images of size (300, 440, 3)
# Y contains vecors of size (11,1) : [screen, books, person close, person + environment, other, relaxed, medium stress, stressed, one point focus, multi point focus, no focus]

#!home/tkal976/.virtualenvs/cv/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import eye_focus as eye_focus
import detect_iris as detect_iris

IMAGE_WIDTH = 440
IMAGE_HEIGHT = 300
CAMERA_NO = 0

#user information
user_no = 100
Gender = 1 # Male = 1 | Female = 2
age = 30
eye_color = 1 # Brown = 1 | Black	= 2 | Blue = 3 | Blue-ish green = 4

def stroop(X,Y,T):
	i = 0
	print("Stroop test is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	t1 = int(round(time.time() * 1000))
	while(int(round(time.time() * 1000)) - t1 < 5000):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(0)
		X.append(frame)
		Y.append(np.transpose([1,0,0,0,0,0,1,0,1,0,0]))
		print(i)
	print("Finished")

def mines(X,Y,T):
	i = 0
	print("Minesweeper is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	t1 = int(round(time.time() * 1000))
	while(int(round(time.time() * 1000)) - t1 < 5000):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(1)
		X.append(frame)
		Y.append(np.transpose([1,0,0,0,0,0,1,0,0,1,0])) #####this is the correct label
		print(i)
	print("Finished")

def tcounting(X,Y,T):
	i = 0
	print("T Counting is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	t1 = int(round(time.time() * 1000))
	while(int(round(time.time() * 1000)) - t1 < 5000):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(2)
		X.append(frame)
		Y.append(np.transpose([1,0,0,0,0,0,1,0,1,0,0]))#####this is the correct label
		print(i)
	print("Finished")

def environment(X,Y,T):
	i = 0
	print("Environment is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(3)
		X.append(frame)
		Y.append(np.transpose([0,0,0,0,1,1,0,0,0,0,1]))
		print(i)
	print("Finished")

def differences(X,Y,T):
	i = 0
	print("Finding differences is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(4)
		X.append(frame)
		Y.append(np.transpose([0,1,0,0,0,0,0,1,0,1,0]))#####this is the correct label
		print(i)
	print("Finished")

def pmaths(X,Y,T):
	i = 0
	print("Simple paper mathematics is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(5)
		X.append(frame)
		Y.append(np.transpose([0,0,0,1,0,0,0,1,0,0,1]))#####this is the correct label but change in h5mod
		print(i)
	print("Finished")

def bmaths(X,Y,T):
	i = 0
	print("Simple mathematics is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(6)
		X.append(frame)
		Y.append(np.transpose([0,1,0,0,0,0,0,1,0,1,0]))#####this is the correct label but change in h5mod
		print(i)
	print("Finished")

def talking(X,Y,T):
	i = 0
	print("Talking is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(7)
		X.append(frame)
		Y.append(np.transpose([0,0,1,0,0,1,0,0,0,0,1]))#####this is the correct label 
		print(i)
	print("Finished")

def youtube(X,Y,T):
	i = 0
	print("YouTube is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	t1 = int(round(time.time() * 1000))
	while(int(round(time.time() * 1000)) - t1 < 5000):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(8)
		X.append(frame)
		Y.append(np.transpose([1,0,0,0,0,1,0,0,0,1,0]))#####this is the correct label
		print(i)
	print("Finished")

def funtext(X,Y,T):
	i = 0
	print("Funny Text is loaded. Press 's' to start recording data")
	while(chr(cv2.waitKey()) != 's'):
		continue
	while True:
		i = i + 1
		ret, frame = cap.read()
		frame  = frame[crop_x1:crop_x2, crop_y1:crop_y2]
		cv2.imshow("eye", frame)
		j = cv2.waitKey(2)
		if(j == 27):
			break
		T.append(9)
		X.append(frame)
		Y.append(np.transpose([0,1,0,0,0,1,0,0,0,1,0]))#####this is the correct label
		print(i)
	print("Finished")

def h5_create(datapath):
	x_shape = (300, 440, 3)
	y_shape = (1,11)
	u_shape = (4,1) #user no, gender, age, eye color
	t_shape = (1,1)
	with h5py.File(datapath, mode='a') as h5f:
		xdset = h5f.create_dataset('X', (0,) + x_shape, maxshape=(None,) + x_shape, dtype='uint8', chunks=(100,) + x_shape)
		ydset = h5f.create_dataset('Y', (0,) + y_shape, maxshape=(None,) + y_shape, dtype='uint8', chunks=(100,) + y_shape)
		udset = h5f.create_dataset('U', (0,) + u_shape, maxshape=(None,) + u_shape, dtype='uint8', chunks=(100,) + u_shape)
		tdset = h5f.create_dataset('T', (0,) + t_shape, maxshape=(None,) + u_shape, dtype='uint8', chunks=(100,) + u_shape)

def h5_append(datapath, U, X, Y, T):
	x_shape = (300, 440, 3)
	y_shape = (1,11)
	u_shape = (1,4) #user no, gender, age, eye color
	t_shape = (1,1)
	with h5py.File(datapath, mode='a') as h5f:
		xdset = h5f['X']
		ydset = h5f['Y']
		udset = h5f['U']
		tdset = h5f['T']
		
		for i in range(X.shape[0]):
			xdset.resize(xdset.shape[0]+1, axis=0)
			xdset[-1:] = X[i]
			print(xdset.shape)
		for i in range(Y.shape[0]):
			ydset.resize(ydset.shape[0]+1, axis=0)
			ydset[-1:] = Y[i]
			print(ydset.shape)
		for i in range(u_shape[0]):
			udset.resize(udset.shape[0]+1, axis=0)
			udset[-1:] = U[i]
			print(udset.shape)
		for i in range(t_shape[0]):
			tdset.resize(tdset.shape[0]+1, axis=0)
			tdset[-1:] = T[i]
			print(tdset.shape)

def crop_and_focus(CAMERA_NO, cap):
	eye_focus.setup_camera(CAMERA_NO, cap)
	x,y = detect_iris.get_cordinates(cap)
	if(x > 480 - 150):
		x = 480 - 150
	if(y < 220):
		y = 220
	return x,y

cap = cv2.VideoCapture(CAMERA_NO)
x_center, y_center = crop_and_focus(CAMERA_NO, cap)
x1 = x_center - int(IMAGE_HEIGHT / 2)
x2 = x_center + int(IMAGE_HEIGHT / 2)
y1 = y_center - int(IMAGE_WIDTH / 2)
y2 = y_center + int(IMAGE_WIDTH / 2)
print([x1,x2,y1,y2])

count1 = 0
while(True):
	count1 = count1 + 1    
	# Capture frame-by-frame
	ret, frame1 = cap.read()
	
	frame  = frame1[ x1:x2 , y1:y2 ]
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# frame_name = str(user_no) + '_' + str(count1) + '.jpg'
	# if(count1 %100 == 0):
	# 	cv2.imwrite(frame_name, frame)
	# cv2.circle(frame, (100, 300), 30, (0, 0, 255), 2)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

X = []
Y = []
T = []

print("Press 's' to start")

while chr(cv2.waitKey()) != 's':
	continue

count = 0

filename = 'tests.h5'

while(True):
	count = count + 1    
	ret, frame = cap.read()
	print("Select Test")

	g = chr(cv2.waitKey())
	filename = str(user_no) + '_' + g + ".h5"

	if(g == '0'):
		stroop(X,Y,T)
	if(g == '1'):
		differences(X,Y,T)
	if(g == '2'):
		mines(X,Y,T)
	if(g == '3'):
		funtext(X,Y,T)
	if(g == '4'):
		environment(X,Y,T)
	if(g == '5'):
		bmaths(X,Y,T)
	if(g == '6'):
		pmaths(X,Y,T)
	if(g == '7'):
		talking(X,Y,T)
	if(g == '8'):
		tcounting(X,Y,T)
	if(g == '9'):
		youtube(X,Y,T)

	if(count == 1):
		break

X_train = np.asarray(X)
Y_train = np.asarray(Y)
U = np.asarray(np.transpose([[user_no,Gender,age,eye_color]])) #user no, gender, age, eye color
T = np.asarray(T)

h5_create(filename)
h5_append(filename, U, X_train, Y_train, T)

cap.release()
cv2.destroyAllWindows()