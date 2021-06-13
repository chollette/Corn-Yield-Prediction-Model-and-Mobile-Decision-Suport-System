import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(data):
    #label encoding
    le = LabelEncoder()
    data['State'] = le.fit_transform(data['State'])
    return data  

def predict_yield(input, model):
    if type(input) == dict:
        df = pd.DataFrame(input)
    else:
        df = input
    cleanData = clean_data(df)
	# make prediction    
    prediction = model.predict(cleanData)
    return prediction
    