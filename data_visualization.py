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

hardness_btn = st.button("Distrubution of Hardness")
if hardness_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Hardness'])
    st.pyplot(fig)

solids_btn = st.button("Distrubution of Solids")
if solids_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Solids'])
    st.pyplot(fig)

chloramines_btn = st.button("Distrubution of Chloramines")
if chloramines_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Chloramines'])
    st.pyplot(fig)

sulfate_btn = st.button("Distrubution of Sulfate")
if sulfate_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Sulfate'])
    st.pyplot(fig)

conductivity_btn = st.button("Distrubution of Conductivity")
if conductivity_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Conductivity'])
    st.pyplot(fig)

toc_btn = st.button("Distrubution of Organic Carbon")
if toc_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Organic_carbon'])
    st.pyplot(fig)


trihalomethanes_btn = st.button("Distrubution of Trihalomethanes")
if trihalomethanes_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Trihalomethanes'])
    st.pyplot(fig)

turbidity_btn = st.button("Distrubution of Turbidity")
if turbidity_btn:
    fig = plt.figure(figsize=(13, 8))
    sns.distplot(df['Turbidity'])
    st.pyplot(fig)

# display the entire dataset using the hist method
df.hist(figsize=(14, 14))
