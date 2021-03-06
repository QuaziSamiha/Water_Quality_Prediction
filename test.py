# import all the required module to train the model or visualize the dataset
# from unicodedata import numeric
import streamlit as st  # pip install streamlit
from streamlit_option_menu import option_menu
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("water_potability.csv")  # loading dataset

with st.sidebar:
    selected = option_menu(
        # menu_title="Select Menu",
        menu_title=None,
        options=['Home', 'Dataset Analysis', 'Dataset Visualization', 'ML Algorithms', 'About App'],
        icons=["droplet", 'bar-chart-line', 'graph-up-arrow', 'code-square', 'clipboard-data']
    )
# ----------------------------------------------------Start Home ---------------------------------------------
if selected == 'Home':
    st.title('WATER QUALITY PREDICTION APP')
    st.markdown('--------------------')
    #  form section
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

    # ----------- POTABILITY Calculation Home Section--------------------------------------
    # fill the null values using the mean value
    df.fillna(df.mean(), inplace=True)
    df.isnull().sum()
    # Train Decision Tree Classifier and Check Accuracy
    X = df.drop("Potability", axis=1)
    Y = df['Potability']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)
    dt = DecisionTreeClassifier(criterion='gini', min_samples_split=10, splitter='best')
    dt.fit(X_train, Y_train)
    prediction = dt.predict(X_test)

    # actions after clicking submit button
    if submitted:
        res = dt.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, toc, trihalomathanes, turbidity]])[0]
        st.write("Potability : ", res)
        if res == 1:
            st.markdown('## Water is drinkable')
        if res == 0:
            st.markdown('## Water is not drinkable')

        st.write('pH:', ph, 'Hardness:', hardness, 'Solids:', solids, 'Chloramines:', chloramines, 'Sulfate:', sulfate)
        st.write('Conductivity:', conductivity, 'Organic Carbon) TOC:', toc, 'Trihalomathanes:', trihalomathanes, 'Turbidity:', turbidity)
# --------------------------------------------------- End Home ----------------------------------------------------

# --------------------------------------------------- Start Dataset Analysis---------------------------------------------------
if selected == "Dataset Analysis":
    st.header(f'{selected} in Different Ways')
    st.markdown('--------------------')

    dt_btn = st.button("Display Whole Dataset")
    if dt_btn:
        st.dataframe(df)

    dtHeadBtn = st.button("Display Top Five Rows of Dataset")
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
        st.write(df.describe())

    potability_btn = st.button("Display the Potability")
    if potability_btn:
        st.write(df.Potability.value_counts())
# --------------------------------------------------- End Dataset Analysis---------------------------------------------------

# --------------------------------------------------Start Dataset Visualization ---------------------------------------------
if selected == "Dataset Visualization":
    st.header(f"{selected} in Different Ways")
    st.markdown('--------------------------')

    potability_btn = st.button('Potability on Dataset')
    if potability_btn:
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x="Potability", data=df)
        st.pyplot(fig)

    boxplot_btn = st.button("Outliers")
    if boxplot_btn:
        fig = plt.figure(figsize=(15, 8))
        sns.boxplot(data=df)
        st.pyplot(fig)

    correlation_btn = st.button("Correlations of Features")
    if correlation_btn:
        fig = plt.figure(figsize=(13, 8))
        sns.heatmap(df.corr(), annot=True, cmap='terrain')
        st.pyplot(fig)

    ph_btn = st.button("Distrubution of pH")
    if ph_btn:
        fig = plt.figure(figsize=(13, 8))
        sns.distplot(df['ph'])
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
# --------------------------------------------------End Dataset Visualization ---------------------------------------------

# --------------------------------------------------Start ML Algo ---------------------------------------------------------
if selected == "ML Algorithms":
    # Replace NULL values with mean value
    df.fillna(df.mean(), inplace=True)
    # divide the dataset in independent and dependent features
    # X contains all the independent features except the potibility
    X = df.drop("Potability", axis=1)
    Y = df['Potability']  # Y contains target feature potability
    # split the dataset into training and testing using train_test_split() function
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)

    algorithm = st.selectbox(
        "Machine Learning Algorithm to Predict Water Quality:",
        ('Select Algorithm', 'Decision Tree Classifier','Naive Bayes Classifier', 'Logistic Regression')
    )

    if algorithm == 'Decision Tree Classifier':
        # train the model using the decision tree classifier
        dt = DecisionTreeClassifier(
            criterion='gini', min_samples_split=10, splitter='best')  # define the DT model
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
        st.text(f"Classification Report = \n\n {classification_report(Y_test, prediction)}")

    if algorithm == 'Logistic Regression':
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        prediction = lr.predict(X_test)

        st.write(f"Training Accuray : {lr.score(X_train, Y_train)*100}")
        st.write(f'Testing Accuracy : {accuracy_score(Y_test,prediction)*100}')
        st.text(f'Confusion Matrix = \n {confusion_matrix(Y_test,prediction)}')
        st.text(f"Classification Report = \n\n {classification_report(Y_test, prediction)}")
# -------------------------------------------------- End ML Algo ---------------------------------------------------------

# ----------------------------------------------------Start About App ---------------------------------------------------
if selected == "About App":
    st.header(f"Details {selected}")
    st.markdown('##### Context: ')
    st.markdown('Access to safe drinking-water is essential to health, basic human right and a component of effective policy for health protection. This is important as a health and developtment issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the intrventions.')
    st.markdown('##### Content: ')
    st.markdown('The water_potability.csv file contains water quality for 3276 different water bodies.')
    st.markdown("##### 1. pH value:")
    st.markdown('pH is an important parameter in evaluating the acid-base balance of water. It is also the indicator of acidic or alkaine condition of water status. WHO has recommended maximum limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52-6.83 which are in the range of WHO standards.')
    st.markdown('##### 2. Hardness:')
    st.markdown('Hardness is mainly caused by calcium and magnesium salts. These salts are dissolbed from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to preciptate soap caused by Calcium and Magnesium.')
    st.markdown('##### 3. Solids (Total Dissolved Solids- TDS):')
    st.markdown('Water has the ability to dissolbe a wide range of inorganic and some organc minerals or salts such an potassium, calcium, sodium, bicarbonates, chlorides, sulfate etc. These minerals produced unwanted taste and color in appearance of water. This is the important parameter for the use of water. The water with high TDS  value indicated that water is highly mineralized. Desirable limit for TDS  is 500 mg/l and maximum limit is 1000 mg/l which prescrived for drinking purpose.')
    st.markdown('##### 4. Chloramines:')
    st.markdown('Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are monst commonly formed when ammonia is added to chlorine to treat drinking waater. Chlorine levels up to 4 mg per liter are considered safe in drinking water.')
    st.markdown('##### 5. Sulfate:')
    st.markdown('Sulfates are naturally occuring substances that are found in minerls, soil and rocks. They are present in ambient air, groundwater plants and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration is seawater is about 2700 mg/L. It ranges from 3 to 30 mg/L in most freshwater suppplies, although mush higher concentrations(1000 mg/L) are found in some geographic locations.')
    st.markdown('##### 6. Conductivity:')
    st.markdown("Pure water is not a good conductor of electric current rather's a good insulator. Increase in icons concentreation enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the iconic process of a solution that enables it to transmit current. According to WHO  standards, EC value should not exceeded 400 microS/cm.")
    st.markdown('##### 7. Organic_carbon:')
    st.markdown('Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2mg/L as TOC in treated/drinking water, and < 4 mg/L in source water which is use for treatment.')
    st.markdown('##### 8. Trihalomethanes:')
    st.markdown('THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.')
    st.markdown('##### 9. Turbidity:')
    st.markdown('The turbidity of water depends on the quality of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genel Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.')
    st.markdown('##### 10. Potability:')
    st.markdown('It indicates if water is safe for human consumption where 1 means Potable and 0 means Not Potable. (0) Water is not safe for drink and (1) Water is safe to drink.')
# ----------------------------------------------------End About App ---------------------------------------------------

