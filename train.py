import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

import os
cnt=0
acc=0
x_train=[]
y_train=[]
for i in range(0,10):
    for j in range(0,50):
        exists = os.path.isfile('dataset/'+str(i)+'/'+str(j)+'.jpg')
        if exists:
                cnt=cnt+1
                gray = cv2.imread('dataset/'+str(i)+'/'+str(j)+'.jpg')
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(255-gray, (28, 28))
                x_train.append(gray)
                y_train.append(i)

x=np.array(x_train)
y=np.array(y_train)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=42)

print(x_train[0])
plt.imshow(x_train[0])
plt.show(1)
print(y_train[0])

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
#print('Number of images in x_test', x_test.shape[0])


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=15)

cnt=0
acc=0
for i in range(0,10):
    for j in range(1,10):
        exists = os.path.isfile('Dataset/i'+str(i)+str(j)+'.jpg')
        if exists:
				cnt=cnt+1
				gray = cv2.imread('Dataset/i'+str(i)+str(j)+'.jpg', 1)
				gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
				gray = cv2.resize(255-gray, (28, 28))
				# gray,thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
				flatten = gray.flatten() / 255.0
				#print(flatten)
				plt.imshow(gray,cmap='Greys')
				plt.show(10)
#
				pred = model.predict(flatten.reshape(1, 28, 28, 1))
				print(pred.argmax())
				if pred.argmax()==i:
					acc=acc+1

print(acc/cnt)

gray = cv2.imread('3.jpg', 1)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(255-gray, (28, 28))
# gray,thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
flatten = gray.flatten() / 255.0
pred = model.predict(flatten.reshape(1, 28, 28, 1))
print(pred.argmax())
cnt=0
acc=0
for i in range(len(x_test)):
    cnt=cnt+1
    pred = model.predict(flatten.reshape(1, 28, 28, 1))
    flatten=x_test[i].reshape(1,28,28,1)
    flatten = x_test[i].astype('float32')
    flatten /=255
    pred = model.predict(flatten.reshape(1, 28, 28, 1))
    if pred.argmax()==y_test[i]:
        acc=acc+1
    #print(pred.argmax())

print(acc/cnt)