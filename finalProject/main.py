from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('./Animals_prediction_model.joblib')

# preprocess data
def preprocess_data(data):
    data = np.array(data)
    data = data / 255.0

    # reshape data to fit the input model
    data = np.reshape(data, (1, 224, 224, 3))
    return data


@app.route('/', methods=['GET'])
def index():
    return "The opening scenes of your prediction"


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = file.read()
    data = preprocess_data(data)
    prediction = model.predict(data)
    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)