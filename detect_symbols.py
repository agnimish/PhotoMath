import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

# code for contours


im = cv2.imread('pic.jpg')

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# im = cv2.filter2D(im, -1, kernel)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(imgray,(10,10))

#plt.imshow(im)
#plt.show(10)

ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

#plt.imshow(thresh)
#plt.show(10)

i=0
_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
list=[]
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if w > 100 and h > 100:
		x=im[y:y + h, x:x + w,:]
		M = cv2.moments(cnt)
		Cx = M['m10'] / M['m00']
		Cy = M['m01'] / M['m00']
		list.append((x,Cx,Cy))
		#cv2.imwrite(str(i)+'.png',x)
	i=i+1


im1=list[1][0]
width = 8
height = 8
dim = (width, height)
# resize image
resized = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
#plt.imshow(resized)
#plt.show(10)



#
#
# # import cv2
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn import datasets, svm, metrics
# # pd.options.display.float_format = '{:,.2f}'.format
# #
# # im = cv2.imread('Img1.jpg')
# # # image = pyplot.imread('img1.jpg')
# #
# # #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# # # plt.imshow(imgray)
# # # plt.show()
# #
# # gblur_im = cv2.GaussianBlur(im, (11,11), 0)
# # plt.imshow(gblur_im)
# # plt.show(10)
# #
# # ret,thresh1 = cv2.threshold(gblur_im,127,255,cv2.THRESH_BINARY)
# # _,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# # for cnt in contours:
# # 	x,y,w,h = cv2.boundingRect(cnt)
# # 	#bound the images
# # 	cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
# #
# # i = 30
# # for cnt in contours:
# # 	x,y,w,h = cv2.boundingRect(cnt)
# # 	#following if statement is to ignore the noises and save the images which are of normal size(character)
# # 	#In order to write more general code, than specifying the dimensions as 100,
# # 	# number of characters should be divided by word dimension
# # 	if (w>30 and h>30):#or(w>100 and h<30):
# # 		#save individual images
# # 		cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
# # 		i=i+1
# #
# # centroids = []
# # for cnt in contours:
# #     M = cv2.moments(cnt)
# #     Cx = M['m10']/M['m00']
# #     Cy = M['m01']/M['m00']
# #     centroids.append((Cx, Cy, cnt))
#
#
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train[0])
plt.imshow(x_train[0])
plt.show(1)
print(y_train[0])
#
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
#plt.imshow(x_train[image_index], cmap='Greys')
#plt.show(10)

print(x_train.shape)

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
model.fit(x=x_train,y=y_train, epochs=2)


#
# image_index = 4444
plt.imshow(x_test[4444].reshape(28, 28),cmap='Greys')
plt.show(10)
# pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred.argmax())
# #plt.imshow(x_test[4444])
import os
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
# digits = datasets.load_digits()
# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
# 	plt.subplot(2, 4, index + 1)
# 	plt.axis('off')
# 	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
# 	plt.title('Training: %i' % label)
# 	plt.show(10)
#
# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
# 	plt.subplot(2, 4, index + 5)
# 	plt.axis('off')
# 	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
# 	plt.title('Prediction: %i' % prediction)
#
# plt.show(10)
#
# img = cv2.imread('5.jpg', 0)
#
# dim = (8, 8)
# # resize image
# resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# img_unrolled = resized_img.ravel()
# value = classifier.predict([img_unrolled])
# print(value)
# plt.imshow(resized_img)
#
# j = 0
# for i in range(4, 11):
#     img = cv2.imread(str(i)+".jpeg", 0)
#     gblur = cv2.GaussianBlur(img, (11,11), 0)
#     ret,thresh1 = cv2.threshold(gblur,127,255,cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         x,y,w,h = cv2.boundingRect(cnt)
#         #bound the images
#         cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
#     for cnt in contours:
#         x,y,w,h = cv2.boundingRect(cnt)
#         #following if statement is to ignore the noises and save the images which are of normal size(character)
#         #In order to write more general code, than specifying the dimensions as 100,
#         # number of characters should be divided by word dimension
#         if (w>30 and h>30):#or(w>100 and h<30):
#             #save individual images
#             cv2.imwrite(str(j)+".jpg",thresh1[y:y+h,x:x+w])
#             j=j+1
#
# import os
#
# images = []
# target = []
# for i in range(0,10):
#     for j in range(1,10):
#         exists = os.path.isfile('i'+str(i)+str(j)+'.jpg')
#         if exists:
#             img = cv2.imread('i'+str(i)+str(j)+'.jpg', 0)
#             resized_img = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
#             img_unrolled = resized_img.ravel()
#             images.append(img_unrolled)
#             target.append(i)
#
# from sklearn.model_selection import train_test_split
# # To apply a classifier on this data, we need to flatten the image, to
# # turn the data in a (samples, feature) matrix:
# # n_samples = len(digits.images)
# # n_samples = len(images)
#
# # data = images.reshape((n_samples, -1))
# # Split Train and Test data
# X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.33, random_state=42)
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
#
# # We learn the digits on the first half of the digits
# classifier.fit(X_train, y_train)
#
# # Now predict the value of the digit on the second half:
# expected = y_test
# predicted = classifier.predict(X_test)
#
# print("Classification report for classifier %s:\n%s\n")
# #       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#
# images_and_predictions = list(zip(images[n_samples // 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#
# plt.show(10)