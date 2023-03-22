import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from matplotlib import pyplot as plt
"""
# Caries Detection AI!

Welcome to Caries Detection AI. A web application that allows you to detect caries from x-ray images.

With Carie Detection, simply upload your x-ray image. Our machine learning algorithms analyze the image and highlighting any caries spots.
The system will then show you the results and details below.
"""

img = st.file_uploader("First!. Please Upload an X-ray Image to detect a caries spot")
colInput, colRes = st.columns(2)
with colInput:
    if (img) :
        st.write("Example image:")
        st.image(img, width=350)
modeler = tf.keras.models.load_model('unet_caries.h5')
modeler.load_weights('test.hdf5')

def predictnow(img):
    #Resizing images, if needed
    SIZE_X = 128 
    SIZE_Y = 128
    train_images = []
    image2 = np.array(img)/255.0
    img3 = cv2.resize(image2, (SIZE_Y, SIZE_X),interpolation=cv2.INTER_CUBIC)
    train_images = np.array(img3)
    train_images = np.expand_dims(train_images, axis=3)
    train_images = tf.keras.utils.normalize(train_images, axis=1)
    test_img_norm=train_images[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (modeler.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    my_cm = plt.cm.get_cmap('gist_rainbow')
    mapped_data = my_cm(predicted_img)
    return  mapped_data


st.write("If you have successfully uploaded the image. Please press the 'Process' button to evaluate.")
clicked = st.button("Process")

if (clicked) :
    if (img) :
        #Tensorflow Graph
        image2 = Image.open(img).convert('RGB')
        resImg = predictnow(image2)

        st.success("✔️ Done!")
        with colRes:
            st.write("Result image: ")
            st.image(resImg, width=350)
        
        with st.expander("About this model"):
            """
            # About This Model
            The architecture is symmetric and consists of two major parts — the left part is called contracting path, 
            which is constituted by the general convolutional process; the right part is expansive path, 
            which is constituted by transposed 2d convolutional layers
            """
            st.write("U-Net architecture that use in this model.")
            st.image("images/unet.png")
            """
            Train a multiclass U-Net architecture by running multiple experiments with different training datasets. 
            In order to verify the performance of the deep recognition model and sub-visualization used. 
            with various training details are as follows
            - The Dataset is 472 image.
            - Input image has dimensions of 128x128x3.
            - Loss function is "binary crossentropy"
            - Optimizer is "Adam Optimizer"
            - The number of training sessions is 50 epoch.
            - Learning rate is 0.001(10^-3)
            - Batch size equal to 16
            """
            st.write("An accuracy and Loss of this AI model")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image("images/Acc.png")
                st.metric("Model Accuracy", "77.14%")
            with col2:
                st.image("images/loss.png")
                st.metric("Model loss", "96.28%")
            
        clicked_reset = st.button("Reset")
        if (clicked_reset):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()
            
   
    else:
        st.warning("Please upload your X-ray image")
        
        
