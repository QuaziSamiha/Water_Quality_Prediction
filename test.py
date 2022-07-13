import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets

st.title('WATER QUALITY PREDICTION APP')
df = pd.read_csv("water_potability.csv")
# df.head()
# st.write(df)
# st.sidebar
st.write(df.head())


# df.shape # (3276, 10) row and column or feature and attribute

# df.isnull().sum()

df.describe()