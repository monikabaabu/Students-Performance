from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('../model/model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours = data['hours']
    attendance = data['attendance']

    prediction = model.predict([[hours, attendance]])

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)