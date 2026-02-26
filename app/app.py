from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

#HOME ROUTE (VERY IMPORTANT)
@app.route('/')
def home():
    return render_template("index.html", result=None)

#PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
        crop_file = request.files['crop_file']
        weather_file = request.files['weather_file']

        #Read files
        if crop_file.filename.endswith('.csv'):
            crop = pd.read_csv(crop_file)
        else:
            crop = pd.read_excel(crop_file)

        if weather_file.filename.endswith('.csv'):
            weather = pd.read_csv(weather_file)
        else:
            weather = pd.read_excel(weather_file)

        #Merge
        data = pd.merge(crop, weather, on="Year")

        #Train model
        X = data[['Rainfall', 'Temperature', 'Humidity']]
        y = data['Yield']

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        #Input
        rainfall = float(request.form['rainfall'])
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])

        prediction = model.predict([[rainfall, temp, humidity]])

        return render_template("index.html", result=round(prediction[0], 2))

    except Exception as e:
        return f"Error: {str(e)}"

#RUN APP
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))