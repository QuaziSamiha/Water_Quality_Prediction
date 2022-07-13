from unicodedata import numeric
import streamlit as st  # pip install streamlit
from streamlit_option_menu import option_menu
import pandas as pd  # pip install pandas
import plotly.express as px  # pip install plotly-express
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("water_potability.csv")

with st.sidebar:
    selected = option_menu(
        # menu_title="Select Menu",
        menu_title=None,
        options=['Home', 'Dataset', 'Papers'],
        icons=["droplet", 'bar-chart-line', 'clipboard-data']
    )

if selected == 'Home':
    st.title('WATER QUALITY PREDICTION APP')
if selected == "Dataset":
    st.title(f'Visualization of {selected} in Different Ways')

    dt_btn = st.button("Whole Dataset")
    if dt_btn:
        st.dataframe(df)

    dtHeadBtn = st.button("First Five Rows of Dataset")
    if dtHeadBtn:
        st.write(df.head())

    shape_btn = st.button("Dataset Size")
    if shape_btn:
        st.write("Total Rows: ", df.shape[0])
        st.write("Total Columns: ", df.shape[1])

    null_btn = st.button("Column's Details")
    if null_btn:
        st.write('Column Name', 'Total Null Value', df.isnull().sum())

    describe_btn = st.button("Describe Dataset")
    if describe_btn:
        st.markdown('------------')
        st.write(df.describe())

    potability_btn = st.button("Potability Rate")
    if potability_btn:
        st.write(df.Potability.value_counts())

        # select_box = st.selectbox(label='Feature', options='numeric_columns')
        # sns.displot(datas[select_box])
        # st.pyplot()

        # sns.countplot(df['Potability'])
        # fig = plt.show()
        # st.area_chart(data=fig, width=0, height=0, use_container_width=True)

if selected == "Papers":
    st.title(f"Read Different {selected} in Water Potability")

df.fillna(df.mean(), inplace=True)
df.isnull().sum()

# Train Decision Tree Classifier and Check Accuracy
X = df.drop("Potability", axis=1)
Y = df['Potability']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)
df = DecisionTreeClassifier(
    criterion='gini', min_samples_split=10, splitter='best')
st.write(df.fit(X_train, Y_train))

prediction = df.predict(X_test)
st.write('Accuracy Score: ', accuracy_score(Y_test, prediction)*100)
st.write('Confusion Matrix: \n', confusion_matrix(Y_test,prediction))
st.write("Classification Report: \n\n", classification_report(Y_test, prediction))

res = df.predict([[5.735724, 158.318745, 25363.016594, 7.728601, 377.543291, 568.304671, 13.626624, 75.952337, 4.732954]])[0]
st.write('res: ', res)
