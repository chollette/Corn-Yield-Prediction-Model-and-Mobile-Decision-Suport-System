import joblib
import sys
import logging
from flask import Flask, request
from flask import jsonify
from outputs.score import predict_yield

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    model_path = './outputs/model.joblib'
    model = joblib.load(model_path)
    result = predict_yield(data, model)
    return jsonify(result)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

