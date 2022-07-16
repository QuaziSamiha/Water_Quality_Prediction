from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from statistics import stdev
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("water_potability.csv")
df.head()

df.shape

df.isnull().sum()

# The fillna() method replaces the NULL values with a specified value.
# Replace NULL values with mean value
df.fillna(df.mean(), inplace=True)
'''
The fillna() method returns a new DataFrame object unless the 
inplace parameter is set to True, in that case the fillna() 
method does the replacing in the original DataFrame instead.
'''
df.isnull().sum()

# how much 0 and how much 1
# resampling data
df.Potability.value_counts()
# value_counts() function to count the frequency of unique values

# visualize the potability using countplot() function of seaborn
sns.countplot(df['Potability'])
plt.show()

# display the ph value using the displot
sns.distplot(df['ph'])  # it's a normal distribution
plt.show()

# display the entire dataset using the hist method
df.hist(figsize=(14, 14))
plt.show()

# visualize the correlation of all the features using heatmap() function
# of seaborn
plt.figure(figsize=(13, 8))
sns.heatmap(df.corr(), annot=True, cmap='terrain')
plt.show()
# we can see there is no correlation between any feature
# so it means we cannot reduce the dimension

# outliers using the boxplot() function
df.boxplot(figsize=(14, 7))

# divide the dataset in independent and dependent features
# X contains all the independent features except the potibility
X = df.drop("Potability", axis=1)
Y = df['Potability']  # Y contains target feature potability

# split the dataset into training and testing using train_test_split() function
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)
# test_size is 20%
# train dataset 80%
# random_state is basically used to ignore the shuffle everytime
# train_test_split() returns 4 datasets

# train the model using the decision tree classifier
dt = DecisionTreeClassifier(
    criterion='gini', min_samples_split=10, splitter='best')  # define the DT model
dt.fit(X_train, Y_train)
# fit the model (it means learn the parameter from training data)

# check the model that how it is performing on the test data
# give the test data and check the accuracy on the test dataset and predicted data set
prediction = dt.predict(X_test)
print(f'Accuracy Score = {accuracy_score(Y_test,prediction)*100}')
print(f'Confusion Matrix = \n {confusion_matrix(Y_test,prediction)}')
print(
    f"Classification Report = \n {classification_report(Y_test, prediction)}")

# predict a single row that how the model perform only one row
res = df.predict([[5.735724, 158.318745, 25363.016594, 7.728601,
                 377.543291, 568.304671, 13.626624, 75.952337, 4.732954]])[0]
res


# apply Hyper Parameter Tuning on DT classifier
model = DecisionTreeClassifier()  # define DT Classifier
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
min_samples_split = [2, 4, 6, 8, 10, 12, 14]
# criterion, splitter, min_samples_split are 3 hyper parameters

# created a dictionary of these parameters
grid = dict(splitter=splitter, criterion=criterion,
            min_samples_split=min_samples_split)
# used repeated stratified k fold cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)  # given 10 splits
# define grid search cv which is basically used to perform hyper parameter tuning
grid_search_dt = GridSearchCV(
    estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
# train the model using the training dataset
grid_search_dt.fit(X_train, Y_train)


# test the model and check the accuracy
print(
    f"Best: {grid_search_dt.best_score_ :.3f } using {grid_search_dt.best_params_}")
means = grid_search_dt.cv_results_["mean_test_score"]
stds = grid_search_dt.cv_results_["std_test_score"]
params = grid_search_dt.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")

print("testing_score:", grid_search_dt.score(X_train, Y_train)*100)
print("Tesing Score:", grid_search_dt.score(X_test, Y_test)*100)
