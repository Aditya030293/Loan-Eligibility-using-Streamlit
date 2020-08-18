## Import libraries:
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor #machine learning model 1
from sklearn.ensemble import RandomForestRegressor #machine learning model 2
from sklearn.metrics import mean_squared_error #regression evaluation
from matplotlib import rcParams #Plotting params.
%matplotlib inline
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("C:/Users/Aditya/Desktop/Kaggle Datasets/Loan Prediction -3/train.csv")
test = pd.read_csv("C:/Users/Aditya/Desktop/Kaggle Datasets/Loan Prediction -3/test.csv")

train['Loan_Status'] = np.where(train['Loan_Status'] == 'Y', 1,0)

df = pd.concat([train,test], axis = 0, ignore_index = True)

df.info()


## MIssing value treatment:

object_col = []

for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <=30:
        object_col.append (column)
        print (f"{column} : {df[column].unique()}")
        print (df[column].value_counts())
        print ("================")

## Fill missing values:
## mode()[0] search?? 
## na not reflect in value_count
        
df['Gender'].value_counts()        
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])       

df['Married'].value_counts()        
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])       

df['Dependents'].value_counts()        
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])       

df['Self_Employed'].value_counts()        
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])       


df['LoanAmount'].describe()        
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())       

df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'].describe()     
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())       


df['Credit_History'].value_counts()
df['Credit_History'].describe()     
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].max())       

df['Credit_History'] = np.where(df['Credit_History'] == 0, 'No','Yes')

df.to_csv("C:/Users/Aditya/Desktop/Kaggle Datasets/Loan Prediction -3/Loan_Clean.csv",
          index = False)




## Encoding Categoriacal Variable:

encode = ['Gender','Married','Education','Self_Employed','Property_Area',
          'Credit_History']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]


## Data Processing:
df.drop (['Loan_ID','Dependents'],axis=1,inplace=True)        

#Dummifying DF.
#df = pd.get_dummies(df)
df.columns

df_train = df.iloc[:614,:]

df_test = df.iloc[614:,:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Loan_Status', axis = 1),
                                                    df_train['Loan_Status'], test_size =0.30,
                                                    random_state = 101)


import xgboost

classifier=xgboost.XGBClassifier()

## Hyper Parameter Optimization

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
eval_metric = ['auc']

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'eval_metric':eval_metric
        }

from sklearn.model_selection import RandomizedSearchCV


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=classifier,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=10,
            n_jobs = 2,
            verbose = 5, 
            random_state=42)


random_cv.fit(X_train,y_train)

random_cv.best_estimator_
random_cv.best_params_
random_cv.best_score_


clf = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=2, missing=None,
       n_estimators=500, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)


# fit the model with the training data
clf.fit(X_train,y_train)


# Saving the model
import pickle
pickle.dump(clf, open('loan_pred_clf.pkl', 'wb'))







