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

chart_data = pd.DataFrame(
    np.random.randn(100,100),
    columns=['accuracy', 'loss']
)
st.line_chart(chart_data)



st.write(pd.DataFrame({
    'Class': ['caries', 'enamel', 'pulp', 'tooth'],
    'Accuracy': [0.57,0.78,0.80,0.75],
    'Loss' : [1.7,0.5,0.3,0.4]
}))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Caries", "55%", "- loss = 50%")
col2.metric("Enamel", "78%", "- loss = 20%")
col3.metric("Pulp", "80%", "- loss = 20%")
col4.metric("Tooth", "70%", "- loss = 20%")