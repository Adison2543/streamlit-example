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
    my_cm = plt.cm.get_cmap('jet')
    mapped_data = my_cm(predicted_img)
    return  mapped_data


st.write("If you have successfully uploaded the image. Please press the 'Process' button to evaluate.")
clicked = st.button("Process")

if (clicked) :
    if (img) :
        #Tensorflow Graph
        image = Image.open(img).convert('RGB')
        resImg = predictnow(image)
        resImg = Image.open(resImg).convert('RGB')
        resImg.putalpha(45)
        res = Image.alpha_composite(image, resImg)
        

        st.success("✔️ Done!")
        with colRes:
            st.write("Result image:")
            st.image(res, width=350)
        
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
        
        
