import matplotlib.pyplot as plt
import numpy as np

import streamlit as st
import pandas as pd

df_inkjet = pd.read_csv("InkjetDB_preprocessing.csv")


st.write("hello")
st.link_button("google", "https://www.google.com/")
# st.line_chart(df_inkjet)

x = np.arange(-10, 10)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'r')

st.pyplot(fig)

st.image("main_image.jpg")
st.video("https://youtu.be/IfsnYv4Ofgw?si=DcByH_gPpg85Qd8l")
