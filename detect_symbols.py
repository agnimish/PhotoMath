import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
import pandas as pd
import tensorflow as tf
import os
pd.options.display.float_format = '{:,.2f}'.format



## Creating train and test data

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
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.4, random_state=42) ## splitting the data in test and train set

##---------------------------------------------------------------------

## training the data using CNN

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



from keras.models import Sequential  # Importing the required Keras modules containing model and layers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()# Creating a Sequential Model and adding the layers
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=2)

##------------------------------------------------------------------------------------------

## Testing the data in test set

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
				flatten = gray.flatten() / 255.0
				plt.imshow(gray,cmap='Greys')
				plt.show(10)
				pred = model.predict(flatten.reshape(1, 28, 28, 1))
				print(pred.argmax())
				if pred.argmax()==i:
					acc=acc+1

print(acc/cnt)

## COMMENTED code for testing on the symbols created which was pretty good in prediction
# gray = cv2.imread('symbols/8.jpg', 1)
# gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# gray = cv2.resize(255-gray, (28, 28))
# flatten = gray.flatten() / 255.0
# pred = model.predict(flatten.reshape(1, 28, 28, 1))
# print(pred.argmax())

#---------------------------------------------------------------------------------

## creating contours in orignal image

# code for contours


im = cv2.imread('Img4.jpg') ## image to evaluate
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(imgray,(10,10))


#ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) ## cCreating contours
list=[]
i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if w > 30 and h > 30:
		img=im[y-15:y + h+15, x-15:x + w+15:]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(255 - gray, (28, 28))
		M = cv2.moments(cnt)
		Cx = M['m10'] / M['m00']
		Cy = M['m01'] / M['m00']
		small=[]
		small.append(gray)
		small.append(Cx)
		small.append(Cy)
		list.append(small)
		#cv2.imwrite(str(i)+'.png',x)
	i=i+1


# for cnt in contours:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	#bound the images
# 	cv2.rectangle(im,(x-15,y-15),(x+w+15,y+h+15),(0,255,0),3)

## REmove all images in the symbols folder
d='/home/harshit/PycharmProjects/photomath/symbols'
filesToRemove = [os.path.join(d,f) for f in os.listdir()]
for f in filesToRemove:
    os.path.join("/home/harshit/PycharmProjects/photomath/symbols", f)
## ---------------------------------------------------------------

i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	#following if statement is to ignore the noises and save the images which are of normal size(character)
	#In order to write more general code, than specifying the dimensions as 100,
	# number of characters should be divided by word dimension
	if (w>30 and h>30) or (w>100 and h<30):
		#save individual images
		plt.imshow(thresh[y:y+h,x:x+w])
		plt.show(10)
		cv2.imwrite('/home/harshit/PycharmProjects/photomath/symbols/'+str(i)+".jpg",thresh[y-15:y+h+15,x-15:x+w+15])
	i=i+1

## predicts the symbols
for i in list:

	plt.imshow(i[0])
	plt.show(10)
	print(i[0].shape)
	print(i[1])
	print(i[2])
	flatten = i[0].flatten() / 255.0
	pred = model.predict(flatten.reshape(1, 28, 28, 1))
	print(pred.argmax())


## sort the list wrt to the x-coordinates of centroid of symbols
def simpleExpr(list):
	return(sorted(list, key = lambda x: x[1]))     


sorted=simpleExpr(list)

x=[ i[2] for i in sorted]
print(x)
