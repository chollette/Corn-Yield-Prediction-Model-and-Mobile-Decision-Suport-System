#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# ## Data 
# Read in data as pandas dataframe and display first 5 rows
data = pd.read_csv('final-data-edited-nozero-07-01-2023.csv')
# ### Prepare Data
# one hot encode state variable
# One-hot encode categorical features
data = pd.get_dummies(data)
# Drop Duplicate
data2 = data.drop_duplicates()
#drop variables not selected after feature selection
data2= data2.drop(['Average_avg-Temp', 'Average-max-temp', 'Clay'], axis = 1)
# convert to float
data2 = data2.astype('float32')
# prepare target value
# Labels are the target values
labels = np.array(data2['Crop Yield'])
labels = np.reshape(labels, (-1,1))
# Remove the labels from the features
data2= data2.drop('Crop Yield', axis = 1)# axis 1 refers to the columns
# Convert to numpy array
data2 = np.array(data2)


# ### Split Data
# Split the data into training, validatio, and testing sets
train_features, val_test_features, train_labels, val_test_labels = train_test_split(data2, labels, test_size = 0.3,shuffle=True, random_state = 0)
X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, shuffle=True, test_size=0.5, random_state=0)

# ### Model
# ##### DNN64
def create_model(data):
    model = keras.Sequential([
        data, 
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))         
    return model
model64 = create_model(normalizer)
history64 = model64.fit(train_features, train_labels,validation_data=(X_val, Y_val),batch_size=100, verbose=1, epochs=60)


# Save model
MODEL_PATH = "./DNNmodel64"
model64.save(MODEL_PATH)


# ##### DNN16
def create_model(data):
    model = keras.Sequential([
        data, 
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))         
    return model
model16 = create_model(normalizer)
history16 = model16.fit(train_features, train_labels,validation_data=(X_val, Y_val),batch_size=100, verbose=1, epochs=60)

MODEL_PATH = "./DNNmodel16"
model16.save(MODEL_PATH)

# #### Validate models performance through Cross-validation
# 
def create_model(data, unit):
    model = keras.Sequential([
        data, 
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(unit, activation='relu'),
        layers.Dense(1)  
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))         
    return model


units = [64,16]
historys = []
score = []

for i in range(0,10):
    print('*********************************************************************************************')
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    train_features, val_test_features, train_labels, val_test_labels = train_test_split(data2, labels, test_size = 0.3,
                                                                              shuffle=True, random_state = i)
    X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, shuffle=True, test_size=0.5, random_state=i)
    
    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
   
    for unit in units:
        #Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', patience=7,  restore_best_weights=True)
        #rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.01)
        
        model = create_model(normalizer, unit)
        history = model.fit(train_features, train_labels,validation_data=(X_val, Y_val),batch_size=100, verbose=1, epochs=60, callbacks=es)
        historys.append(history)
        score1 = model.evaluate(train_features, train_labels, verbose=1)
        score.append(score1)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    i += 1
#print('Mean MAE: %.3f (%.3f)' % (score.mean(), score.std()) )

layer = []
for i in range(0,10):
    for unit in units:
        layer.append(unit)
        
        
L = pd.DataFrame(layer)
B = pd.DataFrame(score)
C = pd.concat([L, B], axis=1)


C.columns = ["layer", "score"]
C = C.sort_values('layer')

C.to_csv('scorebylayer.csv')




