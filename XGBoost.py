#!/usr/bin/env python
# coding: utf-8

# ## Install and Import XGBoost Library

import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
import xgboost
# Pandas is used for data manipulation
import pandas as pd
import numpy as np
import sklearn
from numpy import absolute
from pandas import read_csv
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split


# #### Data and Preparation
# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv('final-data-edited-nozero.csv')
# One-hot encode categorical features
features = pd.get_dummies(features)
features.replace({False: 0, True: 1}, inplace=True)
features2 = features.drop_duplicates()
# Drop duplicate rows
features2 = features.drop_duplicates()
#Drop features not included after feature selection
features2= features2.drop(['Average_avg-Temp', 'Average-max-temp', 'Clay'], axis = 1)
# Labels are the values we want to predict
labels = np.array(features2['Crop Yield'])
# Remove the labels from the features
# axis 1 refers to the columns
data= features2.drop('Crop Yield', axis = 1)
# Saving feature names for later use
data_list = list(data.columns)
# Convert to numpy array
data = np.array(data)


# #### Data Split
# Split the data into training, validation, and testing sets
train_features, val_test_features, train_labels, val_test_labels = train_test_split(data, labels, test_size = 0.3,                                                                    shuffle=True, random_state = 0)
X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, test_size=0.5)

# #### Model
# RF Regression Model model 
model = XGBRegressor(max_depth = 10, subsample=1, n_estimators = 900,learning_rate=0.1)
model.fit(train_features, train_labels)
# evaluate an xgboost regression model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_test, Y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.6f (%.6f)' % (scores.mean(), scores.std()) )

# #### Save Model
import pickle
filename = 'XG_model.dat'
pickle.dump(model, open(filename, "wb"))

