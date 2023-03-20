import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from matplotlib import pyplot as plt
from keras.utils import normalize
import os


#Resizing images, if needed
SIZE_X = 128 
SIZE_Y = 128

TRAIN_PATH = 'images/'

train_images =      []
directory_path_I =  os.listdir(TRAIN_PATH) 

# make sure file is an image
for img_path in directory_path_I:
    if img_path.endswith(('.jpg')):
        img_file = TRAIN_PATH + img_path
        img = cv2.imread(img_file , 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)

train_images = np.array(train_images)
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

#plt.subplot(233)
#plt.title('input image')
#plt.imshow(img, cmap='jet')
#plt.show()



load_model = keras.models.load_model("unet_caries.h5")
load_model.load_weights('test.hdf5')

#y_pred=load_model.predict(img)
#y_pred_argmax=np.argmax(y_pred, axis=3)



test_img_number = 0
test_img = train_images[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (load_model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.title('pred image')
plt.imshow(predicted_img, cmap='jet')
plt.show()