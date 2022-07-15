import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title('Home')

df = pd.read_csv('water_potability.csv')

with st.form(key='form1', clear_on_submit=False):
    ph = st.number_input(label='pH Value', min_value=0.0)
    hardness = st.number_input(label='Hardness', min_value=0.0)
    solids = st.number_input(label='Solids (TDS)', min_value=0.0)
    chloramines = st.number_input(label='Chloramines', min_value=0.0)
    sulfate = st.number_input(label='Sulfate', min_value=0.0)
    conductivity = st.number_input(label='Conductivity', min_value=0.0)
    toc = st.number_input(label='Organic Carbon (TOC)', min_value=0.0)
    trihalomathanes = st.number_input(label='Trihalomathanes', min_value=0.0)
    turbidity = st.number_input(label='Turbidity', min_value=0.0)
    
    submitted = st.form_submit_button(label='Submit')

# fill the null values using the mean value
df.fillna(df.mean(), inplace=True)
df.isnull().sum()

# Train Decision Tree Classifier and Check Accuracy
X = df.drop("Potability", axis=1)
Y = df['Potability']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)

df = DecisionTreeClassifier(
    criterion='gini', min_samples_split=10, splitter='best')
# st.write(df.fit(X_train, Y_train))
df.fit(X_train, Y_train)
prediction = df.predict(X_test)


if submitted:
    res = df.predict([[ph, hardness, solids, chloramines, sulfate,
                     conductivity, toc, trihalomathanes, turbidity]])[0]
    st.write("Potability : ", res)
    if res == 1:
        st.markdown('## Water is drinkable')
    if res == 0:
        st.markdown('## Water is not drinkable')

    st.write('pH:', ph, 'Hardness:', hardness, 'Solids:', solids,
             'Chloramines:', chloramines, 'Sulfate:', sulfate)
    st.write('Conductivity:', conductivity, 'Organic Carbon) TOC:',
             toc, 'Trihalomathanes:', trihalomathanes, 'Turbidity:', turbidity)


# st.write('Accuracy Score: ', accuracy_score(Y_test, prediction)*100)
# st.write('Confusion Matrix: \n', confusion_matrix(Y_test, prediction))
# st.write("Classification Report: \n\n", classification_report(Y_test, prediction))

# res = df.predict([[ph, hardness, solids, chloramines, sulfate,
#                    conductivity, toc, trihalomathanes, turbidity]])[0]

# res = df.predict([[5.735724, 158.318745, 25363.016594, 7.728601,
#                  377.543291, 568.304671, 13.626624, 75.952337, 4.732954]])[0]
# st.write('res: ', res)
