import requests

url = 'http://127.0.0.1:5050/predict'

predict_dir = './predict/1.jpg'
with open(predict_dir, 'rb') as img:
    response = requests.post(url, files={'file': img})
print(response.json())