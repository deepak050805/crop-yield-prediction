from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    # 🔹 Get uploaded files
    crop_file = request.files['crop_file']
    weather_file = request.files['weather_file']

    # 🔹 Save files temporarily
    crop_path = os.path.join("data", crop_file.filename)
    weather_path = os.path.join("data", weather_file.filename)

    crop_file.save(crop_path)
    weather_file.save(weather_path)

    # 🔹 Load data
    crop = pd.read_csv(crop_path)
    weather = pd.read_csv(weather_path)

    # 🔹 Merge
    data = pd.merge(crop, weather, on="Year")

    # 🔹 Train model
    from sklearn.linear_model import LinearRegression

    X = data[['Rainfall', 'Temperature', 'Humidity']]
    y = data['Yield']

    model = LinearRegression()
    model.fit(X, y)

    # 🔹 Predict
    rainfall = float(request.form['rainfall'])
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])

    prediction = model.predict([[rainfall, temp, humidity]])

    return render_template("index.html", result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)