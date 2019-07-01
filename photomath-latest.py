
#%%
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
pd.options.display.float_format = '{:,.2f}'.format
import os


#%%
for i in range(12,13):
    img = cv2.imread("images/"+str(i)+".jpg", 0)
    gblur = cv2.GaussianBlur(img, (11,11), 0)
    ret,thresh1 = cv2.threshold(gblur,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #bound the images
        cv2.rectangle(img,(x-15,y-15),(x+w+15,y+h+15),(0,255,0),3)
    j = 1
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #following if statement is to ignore the noises and save the images which are of normal size(character)
        #In order to write more general code, than specifying the dimensions as 100,
        # number of characters should be divided by word dimension
        if (w>40 and h>40)or(w>100 and h<30): 
            #save individual images
            cv2.imwrite("new_dataset/"+str(i)+"/"+str(j)+".jpg",thresh1[y-15:y+h+15,x-15:x+w+15])
            j=j+1


#%%
# To make seperate contour images inti new dataset
for i in range(13,14):
    exists = os.path.isfile("images/"+str(i)+".jpg")
    print("image exists:", exists)
    if exists:
        img = cv2.imread("images/"+str(i)+".jpg", 0)
        gblur = cv2.GaussianBlur(img, (11,11), 0)
        ret,thresh1 = cv2.threshold(gblur,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        j = 0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            #bound the images
            cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),3)
            if (w>10 and h>10):#or(w>100 and h<30): 
                #save individual images
                cv2.imwrite("new_dataset/"+str(i)+"/"+str(j)+".jpg",thresh1[y-10:y+h+10,x-10:x+w+10])
                j=j+1


#%%

def extract_contours(img):
    gblur = cv2.GaussianBlur(img, (11,11), 0)
    ret,thresh1 = cv2.threshold(gblur,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


#%%
def make_expression(contour):
    for cnt in contour:
        


#%%
j = 1
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #bound the images
    cv2.rectangle(img,(x-15,y-15),(x+w+15,y+h+15),(0,255,0),3)
    #following if statement is to ignore the noises and save the images which are of normal size(character)
    #In order to write more general code, than specifying the dimensions as 100,
    # number of characters should be divided by word dimension
    if (w>50 and h>50):#or(w>100 and h<30): 
        #save individual images
        cv2.imwrite(str(j)+".jpg",thresh1[y-15:y+h+15,x-15:x+w+15])
        j=j+1


#%%
def get_centroids():
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        Cx = M['m10']/M['m00']
        Cy = M['m01']/M['m00']
        centroids.append((Cx, Cy, cnt))
    return centroids    


#%%



#%%
import os

X = []
Y = []
for i in range(0,10):
    for j in range(1,60):
        exists = os.path.isfile('new_dataset/'+str(i)+'/'+str(j)+'.jpg')
        if exists:
            img = cv2.imread('new_dataset/'+str(i)+'/'+str(j)+'.jpg', 0)
            resized_img = cv2.resize(255-img, (28,28), interpolation=cv2.INTER_AREA)
            img_unrolled = resized_img.ravel()
            X.append(img_unrolled)
            Y.append(i)

X = np.array(X)
Y = np.array(Y)
X.shape


#%%


#%% [markdown]
# ###                        PREDICTION USING CONVOLUTION NEURAL NETWORK

#%%

# PREPROCESSING

## 1. Splitting our complete Dataset into three sets:-
# training set, a validation set and a test set.

from sklearn.model_selection import train_test_split

# train = 70% and val_and_test size will be 30% of the overall dataset
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# size of val and test each will be 50% of val_test i.e 15% of whole dataset
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5, random_state=42)


#%%
# 2. Reshaping and Normalisation

#reshape data to fit model
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  

input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
X_test /= 255


#%%
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential([
    Conv2D(28, kernel_size=(3,3), input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(10,activation=tf.nn.softmax)
])


#%%
# Configuring the model with these settings

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%%
# Training on Data

hist = model.fit(X_train, Y_train, # data we are training on
                batch_size=10, epochs=15, # specify the size of our mini-batch and how long we want to train it for (epochs)
                validation_data=(X_val, Y_val)) # model will tell us how we are doing on the validation data 

# At this point, we can experiment with our model(changing hyperparametrs) to check accuracy.


#%%
# Testing our model on test-set

#  index 1 after the model.evaluate function is because
#  the function returns the loss as the first element and the accuracy as the second element.

score = model.evaluate(X_test, Y_test, verbose=0) # second element

print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%
print(X_test.shape)
print(Y_test.shape)


#%%
# # Testing on Real World Images

# img = cv2.imread('6.jpg', 0)
# resized_img = cv2.resize(255-img, (28,28), interpolation=cv2.INTER_AREA)
# img_unrolled = resized_img.ravel()
# X = [img_unrolled] / 255
# Y = [6]

# X = X.reshape(X.shape[0], 28, 28, 1)
# X = X.astype('float32')

# print(X_test.shape)
# print(Y_test.shape)


#%%


#%% [markdown]
# ### PREDICTION USING ARTIFICIAL NEURAL NETWORK

#%%
# PRE-PROCESSING FOR ANN

from sklearn import preprocessing

# Scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

## 4. Splitting our complete Dataset into three sets:-
# training set, a validation set and a test set.

from sklearn.model_selection import train_test_split

# train = 70% and val_and_test size will be 30% of the overall dataset
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

# size of val and test each will be 50% of val_test i.e 15% of whole dataset
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


#%%

#### BUILDING AND TRAINING OUR NEURAL NETWORK

## 1. First Step: Setting up the Architecture

from keras.models import Sequential
from keras.layers import Dense

# Specifying our Architecture(model) Sequentially
#  ‘Dense’ refers to a fully-connected layer

# Hidden layer 1: 50 neurons, ReLU activation
# Hidden layer 2: 50 neurons, ReLU activation
# Output Layer: 1 neuron, Sigmoid activation

model = Sequential([
    Dense(50, activation='relu', input_shape=(784,)), # Hidden layer-1
    Dense(50, activation='relu'), # Hidden layer-2
    Dense(10, activation='sigmoid'),
])

## 2. Second Step: Filling in the best numbers

# Configuring the model with these settings

model.compile(optimizer='sgd', # sgd = stochastic gradient descent
              loss='sparse_categorical_crossentropy', # The loss function for outputs that take the values 0 to 10
              metrics=['accuracy']) # to track accuracy on top of the loss function


# Training on Data

hist = model.fit(X_train, Y_train, # data we are training on
                batch_size=10, epochs=50, # specify the size of our mini-batch and how long we want to train it for (epochs)
                validation_data=(X_val, Y_val)) # model will tell us how we are doing on the validation data 

# At this point, we can experiment with our model(changing hyperparametrs) to check accuracy.

# Testing our model on test-set

#  index 1 after the model.evaluate function is because
#  the function returns the loss as the first element and the accuracy as the second element.

model.evaluate(X_test, Y_test)[1] # second element


