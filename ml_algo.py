import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

st.title("ML Algorithms")
st.markdown("--------------")

df = pd.read_csv("water_potability.csv")

# Replace NULL values with mean value
df.fillna(df.mean(), inplace=True)

# divide the dataset in independent and dependent features
# X contains all the independent features except the potibility
X = df.drop("Potability", axis='columns')
Y = df['Potability']  # Y contains target feature potability

# split the dataset into training and testing using train_test_split() function
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)

algorithm = st.selectbox(
    "Machine Learning Algorithm to Predict Water Quality:",
    ('Select Algorithm', 'Decision Tree Classifier',
     'Naive Bayes Classifier', 'Logistic Regression')
)

if algorithm == 'Decision Tree Classifier':
    # train the model using the decision tree classifier
    dt = DecisionTreeClassifier(criterion='gini', min_samples_split=10, splitter='best')  # define the DT model
    dt.fit(X_train, Y_train)
    # give the test data and check the accuracy on the test dataset and predicted data set
    prediction = dt.predict(X_test)
    st.write(f"Training Accuray : {dt.score(X_train, Y_train)*100}")
    st.write(f'Testing Accuracy : {accuracy_score(Y_test,prediction)*100}')
    st.text(f'Confusion Matrix = \n {confusion_matrix(Y_test,prediction)}')
    st.text(f"Classification Report = \n\n {classification_report(Y_test, prediction)}")

if algorithm == 'Naive Bayes Classifier':
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    prediction = gnb.predict(X_test)

    st.write(f"Training Accuray : {gnb.score(X_train, Y_train)*100}")
    st.write(f'Testing Accuracy : {accuracy_score(Y_test,prediction)*100}')
    st.text(f'Confusion Matrix = \n {confusion_matrix(Y_test,prediction)}')
    st.text(
        f"Classification Report = \n\n {classification_report(Y_test, prediction)}")

if algorithm == 'Logistic Regression':
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    prediction = lr.predict(X_test)

    st.write(f"Training Accuray : {lr.score(X_train, Y_train)*100}")
    st.write(f'Testing Accuracy : {accuracy_score(Y_test,prediction)*100}')
    st.text(f'Confusion Matrix = \n {confusion_matrix(Y_test,prediction)}')
    st.text(
        f"Classification Report = \n\n {classification_report(Y_test, prediction)}")
