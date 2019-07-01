
#%%
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
pd.options.display.float_format = '{:,.2f}'.format


#%%
im = cv2.imread('images/5.jpg',0)
# image = pyplot.imread('img1.jpg')


#%%
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# plt.imshow(imgray)
# plt.show()


#%%
gblur_im = cv2.GaussianBlur(im, (11,11), 0)
plt.imshow(gblur_im)
plt.show()


#%%
ret,thresh1 = cv2.threshold(gblur_im,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #bound the images
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)


#%%
i = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #following if statement is to ignore the noises and save the images which are of normal size(character)
    #In order to write more general code, than specifying the dimensions as 100,
    # number of characters should be divided by word dimension
    if (w>50 and h>50):#or(w>100 and h<30): 
        #save individual images
        cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
        i=i+1


#%%

centroids = []
for cnt in contours:
    M = cv2.moments(cnt)
    Cx = M['m10']/M['m00']
    Cy = M['m01']/M['m00']
    centroids.append((Cx, Cy, cnt))


#%%
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


#%%
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


#%%
img = cv2.imread('5.jpg', 0)

dim = (8, 8)
# resize image
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
img_unrolled = resized_img.ravel()
value = classifier.predict([img_unrolled])
print(value)
plt.imshow(resized_img)


#%%
for i in range(1,2):
    img = cv2.imread("images/"+str(i)+".jpg", 0)
    gblur = cv2.GaussianBlur(img, (11,11), 0)
    ret,thresh1 = cv2.threshold(gblur,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #bound the images
        cv2.rectangle(im,(x-15,y-15),(x+w+15,y+h+15),(0,255,0),3)
    j = 1
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #following if statement is to ignore the noises and save the images which are of normal size(character)
        #In order to write more general code, than specifying the dimensions as 100,
        # number of characters should be divided by word dimension
        if (w>20 and h>50):#or(w>100 and h<30): 
            #save individual images
            cv2.imwrite("new_dataset/"+str(i)+"/"+str(j)+".jpg",thresh1[y-15:y+h+15,x-15:x+w+15])
            j=j+1


#%%
import os

images = []
target = []
for i in range(0,10):
    for j in range(1,10):
        exists = os.path.isfile('i'+str(i)+str(j)+'.jpg')
        if exists:
            img = cv2.imread('i'+str(i)+str(j)+'.jpg', 0)
            resized_img = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
            img_unrolled = resized_img.ravel()
            images.append(img_unrolled)
            target.append(i)


#%%
from sklearn.model_selection import train_test_split
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# n_samples = len(images)

# data = images.reshape((n_samples, -1))
# Split Train and Test data
X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.33, random_state=42)
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
expected = y_test
predicted = classifier.predict(X_test)


#%%
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


#%%
images_and_predictions = list(zip(images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


