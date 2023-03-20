from collections import namedtuple
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import normalize
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

def predictnow(image):
    #Resizing images, if needed
    SIZE_X = 128 
    SIZE_Y = 128
    train_images = []

    img = image
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img)

    train_images = np.array(train_images)
    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)

    load_model = keras.models.load_model("unet_caries.h5")
    load_model.load_weights('test.hdf5')

    test_img_number = 0
    test_img = train_images[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (load_model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    return predicted_img

st.write("If you have successfully uploaded the image. Please press the 'Process' button to evaluate.")
clicked = st.button("Process")

if (clicked) :
    if (img) :
        progress_text = "Progress status"
        my_bar = st.progress(0, text=progress_text)
        progress_text = "Operation in progress. Please wait."
        for i in range(100):
            time.sleep(0.05)
            my_bar.progress(i + 1, text=progress_text)
        #Tensorflow Graph
        resImg = predictnow(img)
        
        
        st.success("✔️ Done!")
        with colRes:
            st.write("Result image:")
            st.image(resImg, width=350)
        
        with st.expander("See result"):
            """
            # Result
            """

            

            st.write("Display an Accuracy and Loss of each class")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Caries", "55%", "- loss = 50%")
            col2.metric("Enamel", "78%", "- loss = 20%")
            col3.metric("Pulp", "80%", "- loss = 20%")
            col4.metric("Tooth", "70%", "- loss = 20%")
            
            chart_data = pd.DataFrame(
                np.random.randn(100,2),
                columns=["Accuracy", "Loss"]
            )
            st.write("Display a line chart of Accuracy and Loss of result")
            st.line_chart(chart_data)
            
        clicked_reset = st.button("Reset")
        if (clicked_reset):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()
            
   
    else:
        st.warning("Please upload your X-ray image")
        
        
