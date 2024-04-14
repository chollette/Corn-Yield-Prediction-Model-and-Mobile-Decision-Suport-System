#!/usr/bin/env python
# coding: utf-8

# install useful libraries
# install flask if not installed
#get_ipython().system('{sys.executable} -m pip install flask')

# install unicorn if not installed
get_ipython().system('{sys.executable} -m pip install gunicorn')

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

# Some important functions
# Encode and serialize data
def prepare_data(data):
    #label encoding
    le = LabelEncoder()
    data['State'] = le.fit_transform(data['State'])
    return data  


def predict(data, model):
    try:
        #load as json file
        data = json.loads(data)['data']  
        data = pd.DataFrame(data)
        #request.get_json()
        
        # make prediction    
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error


#load model saved to disk with pickle for new data
model = pickle.load(open( 'model','rb'))

# prediction on multi-point data points
data = {
        'State': ['Abia', 'Benue', 'Kwara'],
        'Average_avg-Temp': [26.19, 26.93, 25.29],
        'avg-precipitation': [182.66, 119.75, 108.31],
    'avg-windSpeed': [1.46, 1.63, 1.66],
    'PH': [5.45, 5.85, 5.46],
    'Clay': [25.83, 18.83, 27.33],
    'Sand': [68, 51.33, 55.16],
    'Silt': [8.83, 20, 16.83],
    'Hectare': [41.39, 109.76, 63.65]
}

data = pd.DataFrame.from_dict(data)

# Prediction on Single point data 
data = {
        'State': ['Abia'],
        'Average_avg-Temp': [26],
        'avg-precipitation': [230],
    'avg-windSpeed': [1.1],
    'PH': [5],
    'Clay': [18],
    'Sand': [51],
    'Silt': [20],
    'Hectare': [22]
}

data = pd.DataFrame.from_dict(data)

# make a prediction with model
test = prepare_data(data)
# Serializing json   
test = test.to_json()
test = "{\"data\": " + test +"}"
predict(test,model)

