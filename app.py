import pickle
import sys
import logging
from flask import Flask, request
from flask import jsonify
from score import predict_yield

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
	data = request.get_json()
	with open('dfr_model.bin', 'rb') as model_in:
		model = pickle.load(model_in)
		model_in.close()
	result = predict_yield(data, model)
	#return result.tolist()
	return jsonify(result)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.debug = True
    app.run()

