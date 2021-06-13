import pandas as pd
from sklearn.preprocessing import LabelEncoder


def predict_yield(input, model):
    if type(input) == dict:
        df = pd.DataFrame(input)
    else:
        df = input
    #label encoding
    le = LabelEncoder()
    df['State'] = le.fit_transform(df['State'])
    # make prediction    
    prediction = model.predict(df)
    prediction = list(prediction)
    return prediction
    
