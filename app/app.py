from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("models/yield_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", result="")

@app.route('/predict', methods=['POST'])
def predict():
    rainfall = float(request.form['rainfall'])
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])

    prediction = model.predict([[rainfall, temp, humidity]])

    return render_template("index.html", result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)