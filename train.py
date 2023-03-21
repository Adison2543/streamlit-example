# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:44:27 2023

@author: meanie
"""

# -*- coding: utf-8 -*-
"""

@author: USER/
"""

from simple_multi_unet_model import multi_unet_model #Uses softmax 
import tensorflow as tf
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import keras
from matplotlib import pyplot as plt
#from pathlib import Path


#Resizing images, if needed
SIZE_X = 128 
SIZE_Y = 128
n_classes= 80 #Number of classes for segmentation


TRAIN_PATH = 'data_pics/new data/images/'
MASK_PATH = 'data_pics/new data/masks/'


train_images =      []
directory_path_I =  os.listdir(TRAIN_PATH)  



# make sure file is an image
for img_path in directory_path_I:
    if img_path.endswith(('.jpg')):
        img_file = TRAIN_PATH + img_path
        img = cv2.imread(img_file , 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
    

#Convert list to array for machine learning processing        
train_images = np.array(train_images)



train_masks = [] 
directory_path_M = os.listdir(MASK_PATH)

for mask_path in directory_path_M:
   # make sure file is an image
   if mask_path.endswith(('.jpg')):
      mask_file = MASK_PATH + mask_path
      mask = cv2.imread(mask_file, 0)       
      mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
      train_masks.append(mask)


#Convert list to array for machine learning processing        
train_masks= np.array(train_masks)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)


#Further split training data t a smaller subset for quick testing of models

X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.1, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))



###############################################################

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight( class_weight ='balanced', classes = np.unique(train_masks_reshaped_encoded), y = train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)



IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()






#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')
def trainModel():
    res = model.fit(X_train, y_train_cat, 
                        batch_size = 16, 
                        verbose=1, 
                        epochs=100, 
                        validation_data=(X_test, y_test_cat), 
                        #class_weight=class_weights,
                        shuffle=False)
    model.save('test.hdf5')
    return res

history = trainModel()
'''
#------learning rate--------------------------
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

#--------------------------------------------------------------------------
'''
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
#loss, acc = model.evaluate(X_test, y_test_cat)
#print("Accuracy is = ", (acc * 100.0), "%")
#print("loss is = ", (loss * 100.0), "%")
#print("Learning rate is = ", ())



'''
###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''

##################################
#model = get_model()
model.load_weights('test.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  


#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

##################################################

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 80
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#plt.imshow(train_images[0, :,:,0], cmap='gray')
#plt.imshow(train_masks[0], cmap='gray')
#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('???.hdf5')  
def predicted(img):
    prediction = (model.predict(img))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    return predicted_img
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

'''
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

model.save("unet_caries.h5")
'''