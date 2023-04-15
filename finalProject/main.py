from flask import Flask, request, jsonify
import numpy as np
import joblib
import cv2
from keras.models import load_model

app = Flask(__name__)

model = load_model('./Animals_prediction_model.h5')

# preprocess data
def preprocess_data(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    return img


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