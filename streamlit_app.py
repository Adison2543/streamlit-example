from collections import namedtuple
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
"""
# Caries Detection AI!

Welcome to Caries Detection AI. A web application that allows you to detect caries from x-ray images.

With Carie Detection, simply upload your x-ray image. Our machine learning algorithms analyze the image and highlighting any caries spots.
The system will then show you the results and details below.
"""

img = st.file_uploader("First!. Please Upload an X-ray Image to detect a caries spot")

st.write("If you have successfully uploaded the image. Please press the 'Process' button to evaluate.")
clicked = st.button("Process")
progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)
if (clicked) :
    for i in range(100):
        time.sleep(0.1)
        my_bar.progress(i + 1, text=progress_text)
           


with st.expander("See result"):
    """
    # Result
    """
    
    
    chart_data = pd.DataFrame(
        np.random.randn(100,2),
        columns=["Accuracy", "Loss"]
    )
    st.write("Display a line chart of Accuracy and Loss of result")
    st.line_chart(chart_data)


    st.image("https://cdn.discordapp.com/attachments/886148973386162196/1080421305893011476/resEx.png")


    st.write("Display a Accuracy and Loss of each class")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Caries", "55%", "- loss = 50%")
    col2.metric("Enamel", "78%", "- loss = 20%")
    col3.metric("Pulp", "80%", "- loss = 20%")
    col4.metric("Tooth", "70%", "- loss = 20%")