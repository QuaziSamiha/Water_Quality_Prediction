import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn import datasets

# st.title('WATER QUALITY PREDICTION APP')
df = pd.read_csv("water_potability.csv")
# df.head()
# st.write(df)
# st.sidebar

with st.sidebar:
    selected = option_menu(
        # menu_title="Select Menu",
        menu_title=None,
        options=['Home', 'Dataset', 'Contact'],
        icons=["droplet", 'bar-chart-line', 'clipboard-data']
    )

# if selected == 'Select Menu':
#     st.title("Water Quality Prediction App")
if selected == 'Home':
    st.title('WATER QUALITY PREDICTION APP')
if selected == "Dataset":
    st.title(f'Details About {selected}')
    dtHeadBtn = st.button("Show First 5 Rows of Dataset")
# st.write(df.head())
    if dtHeadBtn:
        st.write(df.head())
