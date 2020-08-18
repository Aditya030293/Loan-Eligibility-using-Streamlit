## Import libraries:
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor #machine learning model 1
from sklearn.ensemble import RandomForestRegressor #machine learning model 2
from sklearn.metrics import mean_squared_error #regression evaluation
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Loan Prediction App
This app predicts the **Loan Defaulter** type!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    Gender = st.sidebar.selectbox('Gender',('Male','Female'))
    Married = st.sidebar.selectbox('Married',('Yes','No'))
    #Dependents = st.sidebar.slider('Dependents',1,3,2)
    Education = st.sidebar.selectbox('Education',('Graduate', 'Not_Graduate'))
    Self_Employed = st.sidebar.selectbox('Self_Employed',('Yes','No'))
    ApplicantIncome = st.sidebar.slider('ApplicantIncome',1000,100000,5000)
    CoapplicantIncome = st.sidebar.slider('CoapplicantIncome',0,100000,5000)
    LoanAmount = st.sidebar.slider('LoanAmount',100,1000,500000)
    Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term',0,600,360)
    Credit_History = st.sidebar.selectbox('Credit_History',('Yes','No'))
    Property_Area = st.sidebar.selectbox('Property_Area',('Urban','Rural','Semiurban'))

    data = {'Gender': Gender,
            'Married': Married,
            #'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area}
    
    features = pd.DataFrame(data, index=[0])  
    return features


df_new = user_input_features()

df= pd.read_csv("C:/Users/Aditya/Desktop/Kaggle Datasets/Loan Prediction -3/Loan_Clean.csv")

df = df.drop(columns=['Loan_Status','Loan_ID','Dependents'])

df = pd.concat([df_new,df], axis=0, ignore_index=True)

df.columns

## Encoding Categoriacal Variable:

encode = ['Gender','Married','Education','Self_Employed','Property_Area',
          'Credit_History']


for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1] # Selects only the first row (the user input data)


# Displays the user input features
st.subheader('User Input features')
st.write(df)


# Reads in saved classification model
import pickle
load_clf = pickle.load(open('C:/Users/Aditya/Desktop/Loan/loan_pred_clf.pkl', 'rb'))


# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
prediction_proba_1 = pd.DataFrame(prediction_proba, columns = ['No','Yes'])


st.subheader('Loan Default Status : ')
#Loan_status = np.array(['Yes','No'])
Loan_status = np.where (prediction_proba_1['Yes'] >= 0.5,"Yes","No")
st.write(Loan_status)


# st.subheader('Prediction Probability')
# st.write(prediction_proba)












