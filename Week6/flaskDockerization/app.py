from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return '{"message": "This is a GET function"}'
    elif request.method == 'POST':
        return '{"message": "This is a POST function"}'
    

@app.route('/users', methods=['GET', 'POST'])
def users():
    users = [
        {
            "name": "Charles Macharia",
            "age": 24,
            "city": "Nanyuki, Kenya"
        },

        {
            "name": "Polycarp King'ori",
            "age": 23,
            "city": "Nyahururu, Kenya"
        },

         {
            "name": "Brian Omollo",
            "age": 25,
            "city": "Homabay, Kenya"
        }
    ]

    return users


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=False)
    