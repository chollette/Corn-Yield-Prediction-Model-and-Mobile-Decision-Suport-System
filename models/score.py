import pandas as pd
from sklearn.preprocessing import LabelEncoder


def predict_yield(input, model):
    if type(input) == dict:
        df = pd.DataFrame(input)
    else:
        df = input
    #label one hot encoding
    df = pd.get_dummies(df)
    df.replace({False: 0, True: 1}, inplace=True)
    # make prediction    
    prediction = model.predict(df)
    prediction = list(prediction)
    return prediction
    
