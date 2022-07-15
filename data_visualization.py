import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns

# data_visualization.py
st.title("Data Visualization")
st.markdown('----------------------')

df = pd.read_csv("water_potability.csv")

# visualize the potability using countplot() function of seaborn
# sns.countplot(df['Potability'])
# plt.show()
potability_btn = st.button('Potability on Dataset')
if potability_btn:
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x="Potability", data=df)
    st.pyplot(fig)

# outliers using the boxplot() function
# df.boxplot(figsize=(14, 7))
boxplot_btn = st.button("Outliers")
if boxplot_btn:
    fig = plt.figure(figsize=(15, 8))
    sns.boxplot(data=df)
    st.pyplot(fig)

correlation_btn = st.button("Correlations of Features")
if correlation_btn:
    # visualize the correlation of all the features using heatmap() function
    # of seaborn
    fig = plt.figure(figsize=(13, 8))
    sns.heatmap(df.corr(), annot=True, cmap='terrain')
    # plt.show()
    st.pyplot(fig)
    # we can see there is no correlation between any feature
    # so it means we cannot reduce the dimension

ph_btn = st.button("Distrubution of pH")
if ph_btn:
    # display the ph value using the displot
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['ph'])  # it's a normal distribution
    st.pyplot(fig)


# display the entire dataset using the hist method
# fig = plt.figure(figsize=(13, 8))
df.hist(figsize=(14, 14))
fig = plt.figure(figsize=(10, 4))
sns.distplot(data=df)
st.pyplot(fig)
