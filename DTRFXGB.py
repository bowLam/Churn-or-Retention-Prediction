# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:51:35 2022

@author: Lam Pui King
"""

# =============================================================================
# PACKAGE PREPARE
# =============================================================================

# conda pip install graphviz
# Packages / libraries
import os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
os.chdir(r"C:\Users\User\Desktop\Machine Learning\DTRFXGBoost")
curr=os.getcwd()

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

from pandas import to_datetime
import itertools
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


# =============================================================================
# DATA PREPARATION, INSPECTION AND PREPOSSING
# =============================================================================

raw_data = curr+'\\churn raw data.csv'
# Loading the data
raw_data = pd.read_csv(raw_data, encoding='latin-1')

print(raw_data.shape)

#runs the first 5 rows
raw_data.head()

for i in raw_data:
    unique_value = np.unique(raw_data[i])
    numbofval = len(unique_value)
    if numbofval < 12:
        print('The number of values for feature {} :{} -- {}'.format(i, numbofval,unique_value))
    else:
        print('The number of values for feature {} :{}'.format(i, numbofval))  

# Checking for null values
raw_data.isnull().sum()

raw_data.columns


# Limiting the data
raw_data2 = raw_data[['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']]

# Visualize the data using seaborn Pairplots
g = sns.pairplot(raw_data2, hue = 'Exited', diag_kws={'bw': 0.2})


# # Investigate all the features by our y
# 
# features = ['Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard',
#        'IsActiveMember']
# 
# 
# for f in features:
#     plt.figure()
#     ax = sns.countplot(x=f, data=raw_data2, hue = 'Exited', palette="Set1")

# Making categorical variables into numeric representation

new_raw_data = pd.get_dummies(raw_data2, columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])

new_raw_data.head()

new_raw_data['Balance'].max()

# Scaling our columns
from sklearn import preprocessing
scale_vars = ['CreditScore','EstimatedSalary','Balance','Age']
scaler = preprocessing.StandardScaler()
new_raw_data[scale_vars] = scaler.fit_transform(new_raw_data[scale_vars])
new_raw_data.head()

##5. Splitting the Raw Data - Hold-out validation (I will use CV later)

X = new_raw_data.drop('Exited', axis=1).values# Input features (attributes)
y = new_raw_data['Exited'].values # Target vector
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, test_size=0.1, random_state=0)

# =============================================================================
# DECISION TREE
# =============================================================================

dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
dt.fit(X_train, y_train)

import graphviz 

dot_data = tree.export_graphviz(dt, out_file=None, 
    feature_names=new_raw_data.drop('Exited', axis=1).columns,    
    class_names=new_raw_data['Exited'].unique().astype(str),  
    filled=True, rounded=True,  
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

# Calculating FI
final_fi = pd.DataFrame()
for i, column in enumerate(new_raw_data.drop('Exited', axis=1)):
    print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))
    
    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})
    
    try:
        final_fi = pd.concat([final_fi,fi], ignore_index = True)
    except:
        final_fi = fi
        
        
# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending = False).reset_index()            
final_fi

# Accuracy on Train
print("Training Accuracy is: ", dt.score(X_train, y_train))

# Accuracy on Train
print("Testing Accuracy is: ", dt.score(X_test, y_test))

# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = dt.predict(X_test)

# Plotting Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion')

# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
#The true positive rate (TPR, also called sensitivity) is calculated as TP/TP+FN. 
#TPR is the probability that an actual positive will test positive.
TPR = TP/(TP+FN)
print ("The True Positive rate / Recall per class is: ",TPR)

# Precision or positive predictive value
#Precision looks to see how much junk positives got thrown in the mix. 
#If there are no bad positives (those FPs), then the model had 100% precision. 
#The more FPs that get into the mix, the uglier that precision is going to look.
PPV = TP/(TP+FP)
print ("The Precision per class is: ",PPV)

# False positive rate or False alarm rate
#The false positive rate is calculated as FP/FP+TN, where FP is the number of false positives and TN is 
# =============================================================================
# the number of true negatives (FP+TN being the total number of negatives). 
# It’s the probability that a false alarm will be raised: that a positive result will be
# given when the true value is negative.
# =============================================================================
FPR = FP/(FP+TN)
print ("The False Alarm rate per class is: ",FPR)

# False negative rate or Miss Rate
#The false negative rate – also called the miss rate – is the probability that a true positive will be missed by the test. It’s calculated as FN/FN+TP, where FN is the number of false negatives and TP is the number of true positives (FN+TP being the total number of positives).
FNR = FN/(TP+FN)
print ("The Miss Rate rate per class is: ",FNR)

# Classification error
CER = (FP+FN)/(TP+FP+FN+TN)
print ("The Classification error of each class is", CER)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print ("The Accuracy of each class is", ACC)
print("")

##Total averages :
print ("The average Recall is: ",TPR.sum()/2)
print ("The average Precision is: ",PPV.sum()/2)
print ("The average False Alarm is: ",FPR.sum()/2)
print ("The average Miss Rate rate is: ",FNR.sum()/2)
print ("The average Classification error is", CER.sum()/2)
print ("The average Accuracy is", ACC.sum()/2)


# =============================================================================
# Random Forest
# =============================================================================
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)

# Running Random Forest
#My task: Fine tune it and find the combination with the lowest Classification error.
from itertools import product
n_estimators = 100
max_features = [1, 'sqrt', 'log2']
max_depths = [None, 2, 3, 4, 5]
CERR = []
ACCC=[]
max_feature=[]
max_depth = []
for f, d in product(max_features, max_depths): # with product we can iterate through all possible combinations
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                criterion='entropy', 
                                max_features=f, 
                                max_depth=d, 
                                n_jobs=2,
                                random_state=1337)
    rf.fit(X_train, y_train)
    prediction_test = rf.predict(X=X_test)
    print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))
    cm = confusion_matrix(y_test, prediction_test)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    #plt.figure()
    #plot_confusion_matrix(cm_norm, classes=rf.classes_,
    #title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    CER = (FP+FN)/(TP+FP+FN+TN)
    CER_AVG = CER.sum()/2
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC_AVG = ACC.sum()/2
    max_feature.append(f)
    max_depth.append(d)
    CERR.append(CER_AVG)
    ACCC.append(ACC_AVG)
    
df_random_forest = pd.DataFrame({'max_feature': max_feature,'max_depth': max_depth,
                                 'Classification error': CERR,'Accuracy': ACCC}).sort_values(by = ['max_feature','Accuracy'], ascending=False)

df_random_forest.head(1)
rf = RandomForestClassifier(n_estimators=100, 
                            criterion='entropy', 
                            max_features="sqrt", 
                            n_jobs=2,
                            random_state=1337) 
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)
print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_,
title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_test,prediction_test)))


#Use CV but it does not help to improve the result.
# from sklearn.model_selection import cross_val_predict
# prediction_test = cross_val_predict(rf,X, y,cv=100)
# #prediction_test = rf.predict(X=X_test)
# print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y,prediction_test)))
# cm = confusion_matrix(y, prediction_test)
# cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
# plt.figure()
# plot_confusion_matrix(cm_norm, classes=rf.classes_,
# title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y,prediction_test)))


# =============================================================================
# XGBoost
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV
import xgboost

classifier=xgboost.XGBClassifier(tree_method='gpu_hist')

params={
    "learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth":[2,3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,5,7],
    "gamma":[0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.7]}

clf =RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',cv=3,verbose=3)

# fitting it
clf.fit(X,y)

# best parameters
clf.best_params_

# getting the model with the best parameters
print(clf.best_estimator_)

# fiting the model with the best parameters

final_model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.3, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.2, max_delta_step=0, max_depth=4,
              min_child_weight=5,  monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

# fitting it
final_model.fit(X,y)

pred_xgboost = final_model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, pred_xgboost)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)
                      #title = "XGBoost %s using best parameters" % (123))
                      

# =============================================================================
# MODEL DEPLOYMENT
# =============================================================================

# Loading from CSV
unseen_data2 = pd.read_csv(str(curr+'\\new_unseen_data.csv'), encoding='latin-1')

# Limiting the data
unseen_data2 = unseen_data2[['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']]

output = unseen_data2.copy()

unseen_data2.columns
# dummy variables
unseen_data2 = pd.get_dummies(unseen_data2, columns = ['Geography',
       'Gender', 'HasCrCard', 'IsActiveMember'])

#scaling
scale_vars = ['CreditScore','EstimatedSalary','Balance','Age']
unseen_data2[scale_vars] = scaler.fit_transform(unseen_data2[scale_vars])

unseen_data2.head()

#Making predictions

pred_xgboost = final_model.predict(unseen_data2.values)
pred_prob_xgboost = final_model.predict_proba(unseen_data2.values)

pred_xgboost
pred_prob_xgboost

# function to select second column for probabilities
def column(matrix, i):
    return [row[i] for row in matrix]

column(pred_prob_xgboost, 1)

#Joining the raw data witht the predictions
output['Predictions - Churn or Not'] = pred_xgboost
output['Predictions - Probability to Churn'] = column(pred_prob_xgboost, 1)
output['Predictions - Description'] = 'Empty'
output['Predictions - Description'][output['Predictions - Churn or Not'] == 0] = 'Retention'
output['Predictions - Description'][output['Predictions - Churn or Not'] == 1] = 'Churn'
output.head()

#Exporting the data in a CSV
output.to_csv('Churn_Predictions.csv', sep='\t')









