#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 02:34:24 2020

@author: afranazinin
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV

import imblearn
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import altair as alt

#----------------------------------------------------------------------
#---------------------****USER INPUT*****-------------------------
#----------------------------------------------------------------------

st.sidebar.header("Prediction")


input_Loan_Tenure_Year = st.sidebar.text_input("Loan Tenure Year", 16)
input_Years_to_Financial_Freedom = st.sidebar.text_input("Years to Financial Freedom", 5)
input_Total_Income_for_Join_Application = st.sidebar.text_input("Total Income for Join Application", 7000)
input_Number_of_Loan_to_Approve = st.sidebar.text_input("Number of Loan to Approve", 1)
input_Credit_Card_types = st.sidebar.selectbox('Credit Card Types', ('normal', 'platinum', 'gold'))
input_Number_of_Side_Income = st.sidebar.text_input("Number of Side Income", 1)
input_Number_of_Credit_Card_Facility = st.sidebar.text_input("Number of Credit Card Facility", 2)
input_Credit_Card_Exceed_Months = st.sidebar.text_input("Credit Card Exceed Months", 1)
input_Employment_Type = st.sidebar.selectbox("Employment Type", ('employee', 'self employed', 'government', 'employee', 'fresh graduate'))
input_Number_of_Bank_Products = st.sidebar.text_input("Number of Bank Products", 1)

int(input_Loan_Tenure_Year)
int(input_Years_to_Financial_Freedom)
int(input_Total_Income_for_Join_Application)
int(input_Number_of_Loan_to_Approve)
int(input_Number_of_Side_Income)
int(input_Number_of_Credit_Card_Facility)
int(input_Credit_Card_Exceed_Months)
int(input_Number_of_Bank_Products)


#st.write(input_Loan_Tenure_Year) 
#st.write(input_Total_Income_for_Join_Application) 

input_Total_Duration = int(input_Loan_Tenure_Year) + int(input_Years_to_Financial_Freedom)
int(input_Total_Duration)

if int(input_Total_Income_for_Join_Application) <= 9000:
    input_Total_Income_for_Join_Application2 = 1
if int(input_Total_Income_for_Join_Application) > 9001 and int(input_Total_Income_for_Join_Application) <= 15000:
    input_Total_Income_for_Join_Application2 = 0
if int(input_Total_Income_for_Join_Application) > 15001:
    input_Total_Income_for_Join_Application2 = 2

input_Number_of_Loan_to_Approve2 = int(input_Number_of_Loan_to_Approve)-1

if input_Credit_Card_types == 'platinum':
    input_Credit_Card_types2 = 2
if input_Credit_Card_types == 'normal':
    input_Credit_Card_types2 = 1
if input_Credit_Card_types == 'gold':
    input_Credit_Card_types2 = 0

input_Number_of_Side_Income2 = int(input_Number_of_Side_Income)-1
input_Number_of_Credit_Card_Facility2 = int(input_Number_of_Credit_Card_Facility)-2

if input_Employment_Type == 'employer':
    input_Employment_Type2 = 3
if input_Employment_Type == 'self employed':
    input_Employment_Type2 = 1
if input_Employment_Type == 'government':
    input_Employment_Type2 = 4
if input_Employment_Type == 'employee':
    input_Employment_Type2 = 2
if input_Employment_Type == 'fresh graduate':
    input_Employment_Type2 = 0

input_Number_of_Bank_Products2 = int(input_Number_of_Bank_Products)-1
input_Credit_Card_Exceed_Months2 = int(input_Credit_Card_Exceed_Months)-1


new_data = pd.DataFrame({
    'Total_Duration': [input_Total_Duration],
    'Total_Income_for_Join_Application':[input_Total_Income_for_Join_Application2],
    'Number_of_Loan_to_Approve':[input_Number_of_Loan_to_Approve2],
    'Credit_Card_types':[input_Credit_Card_types2],
    'Number_of_Side_Income':[input_Number_of_Side_Income2],
    'Number_of_Credit_Card_Facility':[input_Number_of_Credit_Card_Facility2],
    'Credit_Card_Exceed_Months':[input_Credit_Card_Exceed_Months2],
    'Employment_Type':[input_Employment_Type2],
    'Number_of_Bank_Products':[input_Number_of_Bank_Products2]
})
    
classifier_name = st.sidebar.selectbox("Select clssifier to run prediction", ("Classifier",
                                                                              "Naive Bayes", 
                                                                              "Decision Tree",
                                                                              "Random Forest",
                                                                              "Gradient Boosting",
                                                                              "K-Nearest Neighbour"))

    


st.write("""
# Deployment
by : ...
""")

@st.cache
def load_data(title):
    original_dataset = pd.read_csv(title+".csv")
    return original_dataset

df = load_data("Bank_CS")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

df2 = df.copy()


#-----------------filling missing value---------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from numpy import isnan

#fill missing value in numerical columns
features1 = ['Loan_Amount', 'Loan_Tenure_Year', 'Years_to_Financial_Freedom',
          'Number_of_Credit_Card_Facility', 'Number_of_Properties', 
          'Number_of_Bank_Products', 'Years_for_Property_to_Completion',
          'Number_of_Side_Income', 'Monthly_Salary', 'Total_Sum_of_Loan',
          'Total_Income_for_Join_Application']

#other columns 
features2 = ['Unnamed: 0', 'Unnamed: 0.1', 'Credit_Card_Exceed_Months',
            'Employment_Type', 'More_Than_One_Products', 'Credit_Card_types',
            'Number_of_Dependents', 'Number_of_Loan_to_Approve', 'Property_Type',
            'State', 'Decision', 'Score' ]

X = df[features1]

imputer = KNNImputer(n_neighbors=3)
dfa = imputer.fit_transform(X)
dfa = pd.DataFrame(dfa, columns = features1)
dfb = pd.DataFrame(df, columns = features2)

#combine
df2 = pd.concat([pd.DataFrame(dfa),  pd.DataFrame(dfb) ], axis=1)
#df2.isnull().sum()

#floor the values

df2['Loan_Amount'] = df2['Loan_Amount'].apply(np.floor)
df2['Loan_Tenure_Year'] = df2['Loan_Tenure_Year'].apply(np.floor)
df2['Years_to_Financial_Freedom'] = df2['Years_to_Financial_Freedom'].apply(np.floor)
df2['Number_of_Credit_Card_Facility'] = df2['Number_of_Credit_Card_Facility'].apply(np.floor)
df2['Number_of_Properties'] = df2['Number_of_Properties'].apply(np.floor)
df2['Number_of_Bank_Products'] = df2['Number_of_Bank_Products'].apply(np.floor)
df2['Years_for_Property_to_Completion'] = df2['Years_for_Property_to_Completion'].apply(np.floor)
df2['Number_of_Side_Income'] = df2['Number_of_Side_Income'].apply(np.floor)
df2['Monthly_Salary'] = df2['Monthly_Salary'].apply(np.floor)
df2['Total_Sum_of_Loan'] = df2['Total_Sum_of_Loan'].apply(np.floor)
df2['Total_Income_for_Join_Application'] = df2['Total_Income_for_Join_Application'].apply(np.floor)



df2.groupby(['Employment_Type'])['Credit_Card_types'].agg(pd.Series.mode)

df2['Credit_Card_types'] = df2['Credit_Card_types'].fillna("normal")
df2.head()

df2["Property_Type"] = df2["Property_Type"].fillna( method ='ffill')

df2['State'] = df2['State'].replace("Johor B", "Johor")
df2['State'] = df2['State'].replace("K.L", "Kuala Lumpur")
df2['State'] = df2['State'].replace("P.Pinang", "Penang")
df2['State'] = df2['State'].replace("Pulau Penang", "Penang")
df2['State'] = df2['State'].replace("N.S", "N.Sembilan")
df2['State'] = df2['State'].replace("SWK", "Sarawak")
df2['State'] = df2['State'].replace("Trengganu", "Terengganu")

del df2['Unnamed: 0.1']






#---------show cleaned data------------------------------------------

if st.checkbox('Show cleaned data'):
    st.subheader('Data with NO missing values')
    st.write(df2)


#feature engineering
df2['Total_Duration'] = df2['Loan_Tenure_Year'] + df2['Years_to_Financial_Freedom']

df_raw = df2.copy()
df_raw.to_csv('raw.csv')

df2.to_csv('preprocessing_cluster.csv')

#Discretization (Loan Amount, Monthly Salary, Total Sum of Loan, Total Income for Join Application)
df2['Loan_Amount']=pd.cut(df2['Loan_Amount'],3,labels=['Low Loan Amount','Average Loan Amount','High Loan Amount'])
df2['Monthly_Salary']=pd.cut(df2['Monthly_Salary'],3,labels=['Low Salary','Average Salary','High Salary'])
df2['Total_Sum_of_Loan']=pd.cut(df2['Total_Sum_of_Loan'],3,labels=['Low Sum of Loan','Average Sum of Loan','High Sum of Loan'])
df2['Total_Income_for_Join_Application']=pd.cut(df2['Total_Income_for_Join_Application'],3,labels=['Low Income for Join Application','Average Income for Join Application','High Income for Join Application'])


#----------------------- EDA ------------------------------------------

st.subheader('EDA for Univariate data')

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(df2.dtypes)


#feature = st.selectbox('Bar chart for which columns?', df2[[i for i in list(df2.columns) if i != 'Unnamed:0']]
#st.bar_chart(df2[feature])

#df[[i for i in list(df.columns) if i != '<your column>']]

#feature = st.selectbox('Bar chart for which columns?', df2.columns[:, df2.columns != 'Unnamed: 0'])
#st.bar_chart(df2[feature])

feature = st.selectbox('Bar chart for which columns?', df2.columns[0:23])
st.bar_chart(df2[feature])


#b = st.multiselect('?', cols)

cols = ["Loan_Amount","Loan_Tenure_Year","Years_to_Financial_Freedom",
           "Number_of_Credit_Card_Facility", "Number_of_Properties",
           "Number_of_Bank_Products","Years_for_Property_to_Completion",
           "Number_of_Side_Income", "Monthly_Salary","Total_Sum_of_Loan ",
           "Total_Income_for_Join_Application", "Credit_Card_Exceed_Months",
           "Number_of_Dependents", "Number_of_Loan_to_Approve",
            "Score"]

           
a = pd.DataFrame(df2, columns = cols)
a.hist()
st.pyplot()

#plt.figure(1)
#plt.subplot(121)
#sns.distplot(df2['Total_Duration'](figsize= 2,2))
#st.pyplot()

df2['Total_Duration'].plot.box(figsize=(16,5))
st.pyplot()

st.subheader('EDA for Multivariate data')


#st.write(pd.crosstab(df2["Number_of_Credit_Card_Facility"],df2["Decision"]))



#----------------------SMOTE----------------------------------------

st.subheader('Handling imbalance class')

#apply smote
#indices1 = [0,2,3,5,6,13,15,17,18,19,20]



st.bar_chart(df2["Decision"])

st.write("Before")
st.write(df2["Decision"].value_counts())

#apply smote
indices1 = [0,8,9,10,13,14,15,18,19,20]
smt = imblearn.over_sampling.SMOTENC(categorical_features = indices1,
                                     sampling_strategy="minority",
                                     random_state=42, k_neighbors=5)

X = df2.drop("Decision", 1)
y = df2["Decision"]
features = X.columns

x = pd.DataFrame(X, columns = features)

X_res, y_res = smt.fit_resample(x, y)

st.write("After")
st.write((y_res.value_counts()))

y_res.value_counts().plot(kind="bar")
plt.title("Class Distribution")
X_res = pd.DataFrame(X_res, columns = features)

df3 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
#df3.head()

#floor the values

df3['Loan_Tenure_Year'] = df3['Loan_Tenure_Year'].apply(np.floor)
df3['Years_to_Financial_Freedom'] = df3['Years_to_Financial_Freedom'].apply(np.floor)
df3['Number_of_Credit_Card_Facility'] = df3['Number_of_Credit_Card_Facility'].apply(np.floor)
df3['Number_of_Properties'] = df3['Number_of_Properties'].apply(np.floor)
df3['Number_of_Bank_Products'] = df3['Number_of_Bank_Products'].apply(np.floor)
df3['Years_for_Property_to_Completion'] = df3['Years_for_Property_to_Completion'].apply(np.floor)
df3['Number_of_Side_Income'] = df3['Number_of_Side_Income'].apply(np.floor)

df3.to_csv('preprocessing.csv')





#----------------------------Feature Selection--------------------------

st.subheader("Feature Selection")

preprocessed_data = pd.read_csv('preprocessing.csv')
#preprocessed_data.shape

#Label encoding

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

d = defaultdict(LabelEncoder)

fit = preprocessed_data.apply(lambda x: d[x.name].fit_transform(x))



#Inverse the encoded

fit.apply(lambda x: d[x.name].inverse_transform(x))


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


df7 = fit.copy()
fit.to_csv('encoded.csv')

del df7['Unnamed: 0']
del df7['Unnamed: 0.1']
del df7['Loan_Tenure_Year']
del df7['Years_to_Financial_Freedom']
#del df7['Number_of_Side_Income_log']


X = df7.drop("Decision",1)
y = df7['Decision'].astype('int64')
colnames = X.columns

#st.write(df7.info())

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto", random_state=1)


feat_selector.fit(X.values, y.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=["Features", "Score"])
boruta_score = boruta_score.sort_values("Score", ascending=False)

if st.checkbox('Show Top 10'):
    st.subheader('Top 10')
    st.write(boruta_score.head(10))
    st.pyplot(sns_boruta_plot = sns.catplot(x="Score", y="Features", 
                                            data = boruta_score[0:10], 
                                            kind = "bar", height=14, aspect=1.9, 
                                            palette='coolwarm'))
    
if st.checkbox('Show Bottom 9'):
    st.subheader('Bottom 9')
    st.write(boruta_score.tail(9))
    st.pyplot(sns_boruta_plot = sns.catplot(x="Score", y="Features", 
                                            data = boruta_score[11:19], 
                                            kind = "bar", height=14, aspect=1.9, 
                                            palette='coolwarm'))


#st.pyplot(sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", height=14, aspect=1.9, palette='coolwarm'))

#st.plt.title("Boruta Top 10 Features")





#-------------------------Association Rule Mining-----------------
    
st.subheader("Association Rule Mining")

from apyori import apriori

arm_data = pd.read_csv('preprocessing.csv')

arm_data['Credit_Card_Exceed_Months'] =  'Credit_Card_Exceed_Months_'+arm_data['Credit_Card_Exceed_Months'].apply(str)
arm_data['Employment_Type'] =  'Employment_Type_'+arm_data['Employment_Type'].apply(str)
arm_data['Loan_Amount'] =  'Loan_Amount_'+arm_data['Loan_Amount'].apply(str)
arm_data['Loan_Tenure_Year'] =  'Loan_Tenure_Year_'+arm_data['Loan_Tenure_Year'].apply(str)
arm_data['More_Than_One_Products'] =  'More_Than_One_Products_'+arm_data['More_Than_One_Products'].apply(str)
arm_data['Number_of_Dependents'] =  'Number_of_Dependents_'+arm_data['Number_of_Dependents'].apply(str)
arm_data['Years_to_Financial_Freedom'] =  'Years_to_Financial_Freedom_'+arm_data['Years_to_Financial_Freedom'].apply(str)
arm_data['Property_Type'] =  'Property_Type_'+arm_data['Property_Type'].apply(str)
arm_data['Years_for_Property_to_Completion'] =  'Years_for_Property_to_Completion_'+arm_data['Years_for_Property_to_Completion'].apply(str)
arm_data['State'] =  'State_'+arm_data['State'].apply(str)
arm_data['Number_of_Side_Income'] =  'Number_of_Side_Income_'+arm_data['Number_of_Side_Income'].apply(str)
arm_data['Monthly_Salary'] =  'Monthly_Salary_'+arm_data['Monthly_Salary'].apply(str)
arm_data['Total_Sum_of_Loan'] =  'Total_Sum_of_Loan_'+arm_data['Total_Sum_of_Loan'].apply(str)
arm_data['Total_Income_for_Join_Application'] =  'Total_Income_for_Join_Application_'+arm_data['Total_Income_for_Join_Application'].apply(str)
arm_data['Score'] = 'Score_'+arm_data['Score'].apply(str)
arm_data['Total_Duration'] = 'Total_Duration_'+arm_data['Total_Duration'].apply(str)
arm_data['Credit_Card_types'] = 'Credit_Card_types_'+arm_data['Credit_Card_types'].apply(str)
arm_data['Decision'] = 'Decision_'+arm_data['Decision'].apply(str)
arm_data['Number_of_Credit_Card_Facility'] = 'Number_of_Credit_Card_Facility_'+arm_data['Number_of_Credit_Card_Facility'].apply(str)
arm_data['Number_of_Bank_Products'] = 'Number_of_Bank_Products_'+arm_data['Number_of_Bank_Products'].apply(str)
arm_data['Number_of_Loan_to_Approve'] = 'Number_of_Loan_to_Approve_'+arm_data['Number_of_Loan_to_Approve'].apply(str)

df8 = arm_data[['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']] 

records = []
for i in range(0, 3538):
    records.append([str(df8.values[i,j]) for j in range(0, 9)])
    
association_rules = apriori(records, min_support=0.07, min_confidence=0.2, min_lift=2, min_length=2)
association_results = list(association_rules)

st.write("Total Number of Association Rules :", len(association_results))
#st.write(len(association_results))

if st.checkbox('Show all the Association Rules'):
    cnt =0
    for item in association_results:
        cnt += 1
        # first index of the inner list
        # Contains base item and add item
        #if st.checkbox('Show the Association Rules'):
        pair = item[0] 
        items = [x for x in pair]
        st.write("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])
    
        #second index of the inner list
        st.write("Support: " + str(round(item[1],3)))
    
        #third index of the list located at 0th
        #of the third index of the inner list
    
        st.write("Confidence: " + str(round(item[2][0][2],4)))
        st.write("Lift: " + str(round(item[2][0][3],4)))
        st.write("==========================================================")




#----------------------------------------------------------------------
#---------------------****CLASSIFICATION*****-------------------------
#----------------------------------------------------------------------


st.subheader("Classification") 

#---------------------------Naive Bayes------------------------------
    
st.write("Naive Bayes") 

from sklearn import preprocessing

preprocessed_data2 = pd.read_csv('encoded.csv')

feature_cols = ['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']        

X = preprocessed_data2[feature_cols]
y = preprocessed_data2.Decision

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=2)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

#st.write ("Score : ", nb.score(X_test, y_test))



#----------confusion Matrix(NB)----------------------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

confusion_majority_nb = confusion_matrix(y_test, y_pred)

prob_nb = nb.predict_proba(X_test)
prob_nb = prob_nb[:, 1]

if st.checkbox('Show more details for Naive Bayes'):
    
    confusion_majority_nb = confusion_matrix(y_test, y_pred)
    
    st.write ("Score : ", nb.score(X_test, y_test))
    
    st.write('Accuracy on test set= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    
    #prob_nb = nb.predict_proba(X_test)
    #prob_nb = prob_nb[:, 1]
    
    auc_nb = roc_auc_score(y_test, prob_nb)
    st.write("AUC: %.2f" % auc_nb)
    
    st.write('Mjority classifier Confusion Matrix\n', confusion_majority_nb)
    
    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority_nb[0][0])
    st.write('Mjority FP=', confusion_majority_nb[0][1])
    st.write('Mjority FN= ', confusion_majority_nb[1][0])
    st.write('Mjority TP= ', confusion_majority_nb[1][1])
    st.write('**********************')
    
    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))



    cm_matrix_nb = pd.DataFrame(data=confusion_majority_nb, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots(figsize=(2,2))
    sns.heatmap(cm_matrix_nb, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot()





#---------------------------Decision Tree------------------------------

st.write("Decision Tree")
import pandas as pd
from sklearn import preprocessing # label encoding
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split functionn

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn import metrics

df = pd.read_csv('encoded.csv')

feature_cols = ['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']

X = df[feature_cols] #features
y = df.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=2)

#to see the optimal max depth
max_depth_range = list(range(1, 11))
accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_depth = depth, random_state=2)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    #if st.checkbox("Show Decision Tree's optimal max depth with corrresponding score", key="dt"):
    #st.write('depth',depth,': ', score)
    
    
clf = DecisionTreeClassifier(criterion='entropy', max_depth = 10)

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#----------confusion Matrix(DT)----------------------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

confusion_majority_dt = confusion_matrix(y_test, y_pred)

prob_dt = clf.predict_proba(X_test)
prob_dt = prob_dt[:, 1]

if st.checkbox('Show more details for Decision Tree'):

    st.write('Accuracy on test set= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    
    #prob_dt = clf.predict_proba(X_test)
    #prob_dt = prob_dt[:, 1]
    
    auc_dt = roc_auc_score(y_test, prob_dt)


    st.write("AUC: %.2f" % auc_dt)
    
    st.write('Majority classifier Confusion Matrix\n', confusion_majority_dt)
    
    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority_dt[0][0])
    st.write('Mjority FP=', confusion_majority_dt[0][1])
    st.write('Mjority FN= ', confusion_majority_dt[1][0])
    st.write('Mjority TP= ', confusion_majority_dt[1][1])
    st.write('**********************')
    
    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    


    cm_dt = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=confusion_majority_dt, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots(figsize=(2,2))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Greens')
    st.pyplot()


#---------------------------Random Forest------------------------------

st.write("Random Forest")

import pandas as pd
from sklearn import preprocessing # label encoding
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split # Import train_test_split functionn

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn import metrics

df = pd.read_csv('encoded.csv')

feature_cols = ['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']

X = df[feature_cols] #features
y = df.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=2)

max_depth_range = list(range(1, 11))
accuracy = []
for depth in max_depth_range:
    
    rf = RandomForestClassifier(n_estimators=100,max_depth = depth, 
                                random_state=2)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    
    #if st.checkbox("Show Random Forest's optimal max depth with corresponding score",key="rf"):
    #st.write('depth',depth,': ', score)
    
rf = RandomForestClassifier(n_estimators=100, random_state=2, max_depth=10)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

confusion_majority_rf = confusion_matrix(y_test, y_pred)

prob_rf = rf.predict_proba(X_test)
prob_rf = prob_rf[:, 1]

if st.checkbox('Show more details for Random Forest'):

    
    st.write('Accuracy on test set= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    
    #prob_rf = rf.predict_proba(X_test)
    #prob_rf = prob_rf[:, 1]
    
    auc_rf = roc_auc_score(y_test, prob_rf)
    st.write("AUC: %.2f" % auc_rf)
    
    st.write('Majority classifier Confusion Matrix\n', confusion_majority_rf)
    
    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority_rf[0][0])
    st.write('Mjority FP=', confusion_majority_rf[0][1])
    st.write('Mjority FN= ', confusion_majority_rf[1][0])
    st.write('Mjority TP= ', confusion_majority_rf[1][1])
    st.write('**********************')
    
    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
#----------confusion Matrix(RF)----------------------


    cm_rf = confusion_matrix(y_test, y_pred)

    cm_matrix = pd.DataFrame(data=confusion_majority_rf, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots(figsize=(2,2))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Reds')
    st.pyplot()


#---------------------------Gradient Boosting------------------------------

st.write("Gradient Boosting")

from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('encoded.csv')
feature_cols = ['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']
                
                
                 

X = df[feature_cols] #features
y = df.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=2)

lr = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for i in lr:
    gb = GradientBoostingClassifier(n_estimators=100, max_depth = 2,
                                   learning_rate = i)
    gb.fit(X_train, y_train)
    score = gb.score(X_test, y_test)
    
    #if st.checkbox('Show the learning rate with corresponding score', ):
    #st.write('learning rate ',i,': ', score)


gb = GradientBoostingClassifier(n_estimators=100, max_depth = 2, 
                                learning_rate = 0.75)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

confusion_majority_gb = confusion_matrix(y_test, y_pred)

prob_gb = gb.predict_proba(X_test)
prob_gb = prob_gb[:, 1]

if st.checkbox('Show more details for Gradient Boosting'):

    st.write('Accuracy on test set= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    
    #prob_gb = gb.predict_proba(X_test)
    #prob_gb = prob_gb[:, 1]
    
    auc_gb = roc_auc_score(y_test, prob_gb)

#if st.checkbox('Show more details for Gradient Boosting'):

    
    st.write("AUC: %.2f" % auc_gb)
    
    st.write('Majority classifier Confusion Matrix\n', confusion_majority_gb)
    
    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority_gb[0][0])
    st.write('Mjority FP=', confusion_majority_gb[0][1])
    st.write('Mjority FN= ', confusion_majority_gb[1][0])
    st.write('Mjority TP= ', confusion_majority_gb[1][1])
    st.write('**********************')
    
    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

#----------confusion Matrix(GB)----------------------



    confusion_majority_gb = confusion_matrix(y_test, y_pred)
    
    cm_matrix = pd.DataFrame(data=confusion_majority_gb, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    
    fig, ax = plt.subplots(figsize=(2,2))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='RdPu')
    st.pyplot()


#---------------------------K-Nearest neighbour------------------------------

st.write("K-Nearest Neighbour")

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('encoded.csv')
feature_cols = ['Total_Duration',
                'Total_Income_for_Join_Application',
                'Number_of_Loan_to_Approve',
                'Credit_Card_types',
                'Number_of_Side_Income',
                'Number_of_Credit_Card_Facility',
                'Credit_Card_Exceed_Months',
                'Employment_Type',
                'Number_of_Bank_Products']

X = df[feature_cols] #features
y = df.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=2)

#to see the optimal number of neigbors
neighbor_range = [1,3,5,7,9,11,13,15]
accuracy = []
for n in neighbor_range:
    
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    
    #if st.checkbox('Show the optimal number of neighbours'):
    #st.write('neighbour ',n,': ', score)
    
    
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


#----------confusion Matrix(KNN)----------------------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

confusion_majority_knn = confusion_matrix(y_test, y_pred)

prob_knn = knn.predict_proba(X_test)
prob_knn = prob_knn[:, 1]

if st.checkbox('Show more details for K-Nearest Neighbours'):

    st.write('Accuracy on test set= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    
    #prob_knn = knn.predict_proba(X_test)
    #prob_knn = prob_knn[:, 1]
    
    auc_knn = roc_auc_score(y_test, prob_knn)
    st.write("AUC: %.2f" % auc_knn)
    
    st.write('Majority classifier Confusion Matrix\n', confusion_majority_knn)
    
    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority_knn[0][0])
    st.write('Mjority FP=', confusion_majority_knn[0][1])
    st.write('Mjority FN= ', confusion_majority_knn[1][0])
    st.write('Mjority TP= ', confusion_majority_knn[1][1])
    st.write('**********************')
    
    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))   



    cm_knn = confusion_matrix(y_test, y_pred)
    
    cm_matrix = pd.DataFrame(data=confusion_majority_knn, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    
    
    fig, ax = plt.subplots(figsize=(2,2))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Purples')
    st.pyplot()



#--------------------MODEL EVALUATION (CLASSIFICATION)----------------
#________________________________________________________________

fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_nb)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_dt)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, prob_rf)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, prob_knn)
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, prob_gb)

fig, ax = plt.subplots(figsize=(4,4))

plt.plot(fpr_NB, tpr_NB, color='orange', label='NB') 
plt.plot(fpr_DT, tpr_DT, color='blue', label='DT') 
plt.plot(fpr_rf, tpr_rf, color='red', label='RF')
plt.plot(fpr_knn, tpr_knn, color='black', label='KNN')
plt.plot(fpr_gb, tpr_gb, color='green', label='GB')

plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()








#---------CHOOSE CLASSIFIER ---------#



#if st.sidebar.checkbox('Predict using Naive Bayes'):
if classifier_name == "Naive Bayes":
    predicted = nb.predict(new_data)
    predicted_proba = nb.predict_proba(new_data)

    if int(predicted) == 0:
        predicted2 = 'Accept'
    if int(predicted) == 1:
        predicted2 = 'Reject'
        
    def predicted_nb():
        predicted_accept = predicted_proba[0][0]
        predicted_reject = predicted_proba[0][1]
        
        nb_pred = { 'Accept': predicted_accept,
                    'Reject': predicted_reject
                    }
        nb_pred_df = pd.DataFrame(nb_pred, index=[0])
        return nb_pred_df
    
    st.sidebar.write('Naive Bayes prediction: ', predicted2)
    
    predicted_nb_result = predicted_nb()
    st.sidebar.write(predicted_nb_result)
    #st.bar_chart(predicted_nb_result)
    
    #st.write(alt.Chart(predicted_nb_result))
    
    
    

#if st.sidebar.checkbox('Predict using Decision Tree'):
if classifier_name == "Decision Tree":
    predicted = clf.predict(new_data)
    predicted_proba = clf.predict_proba(new_data)

    if int(predicted) == 0:
        predicted2 = 'Accept'
    if int(predicted) == 1:
        predicted2 = 'Reject'
    
    def predicted_dt():
        predicted_accept = predicted_proba[0][0]
        predicted_reject = predicted_proba[0][1]
        
        dt_pred = { 'Accept': predicted_accept,
                    'Reject': predicted_reject
                    }
        dt_pred_df = pd.DataFrame(dt_pred, index=[0])
        return dt_pred_df
    
    st.sidebar.write('Decision Tree prediction: ', predicted2)
    
    predicted_dt_result = predicted_dt()
    st.sidebar.write(predicted_dt_result)
    
    

#if st.sidebar.checkbox('Predict using Random Forest'):
if classifier_name == "Random Forest":
    predicted = rf.predict(new_data)
    predicted_proba = rf.predict_proba(new_data)

    if int(predicted) == 0:
        predicted2 = 'Accept'
    if int(predicted) == 1:
        predicted2 = 'Reject'
    
    def predicted_rf():
        predicted_accept = predicted_proba[0][0]
        predicted_reject = predicted_proba[0][1]
        
        rf_pred = { 'Accept': predicted_accept,
                    'Reject': predicted_reject
                    }
        rf_pred_df = pd.DataFrame(rf_pred, index=[0])
        return rf_pred_df
    
    st.sidebar.write('Random Forest prediction: ', predicted2)
    
    predicted_rf_result = predicted_rf()
    st.sidebar.write(predicted_rf_result)
    

#if st.sidebar.checkbox('Predict using Gradient Boosting'):
if classifier_name == "Gradient Boosting":
    predicted = gb.predict(new_data)
    predicted_proba = gb.predict_proba(new_data)

    if int(predicted) == 0:
        predicted2 = 'Accept'
    if int(predicted) == 1:
        predicted2 = 'Reject'
    
    def predicted_gb():
        predicted_accept = predicted_proba[0][0]
        predicted_reject = predicted_proba[0][1]
        
        gb_pred = { 'Accept': predicted_accept,
                    'Reject': predicted_reject
                    }
        gb_pred_df = pd.DataFrame(gb_pred, index=[0])
        return gb_pred_df
    
    st.sidebar.write('Gradient Boosting prediction: ', predicted2)
    
    predicted_gb_result = predicted_gb()
    st.sidebar.write(predicted_gb_result)
    
    

#if st.sidebar.checkbox('Predict using KNN'):
if classifier_name == "K-Nearest Neighbour":
    predicted = knn.predict(new_data)
    predicted_proba = knn.predict_proba(new_data)

    if int(predicted) == 0:
        predicted2 = 'Accept'
    if int(predicted) == 1:
        predicted2 = 'Reject'
    
    def predicted_knn():
        predicted_accept = predicted_proba[0][0]
        predicted_reject = predicted_proba[0][1]
        
        knn_pred = { 'Accept': predicted_accept,
                    'Reject': predicted_reject
                    }
        knn_pred_df = pd.DataFrame(knn_pred, index=[0])
        return knn_pred_df
    
    st.sidebar.write('KNN prediction: ', predicted2)
    
    predicted_knn_result = predicted_knn()
    st.sidebar.write(predicted_knn_result)