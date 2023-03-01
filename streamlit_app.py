from collections import namedtuple
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
st.write("\n \n")
chart_data = pd.DataFrame(
    np.random.randn(100,2),
    columns=["Accuracy", "Loss"]
)
st.write("Display a line chart of Accuracy and Loss of result")
st.line_chart(chart_data)

st.write("Display a Accuracy and Loss of each class")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Caries", "55%", "- loss = 50%")
col2.metric("Enamel", "78%", "- loss = 20%")
col3.metric("Pulp", "80%", "- loss = 20%")
col4.metric("Tooth", "70%", "- loss = 20%")