from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "yield_model.pkl")

# Load model (if exists)
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    model = None

# ✅ HOME ROUTE (IMPORTANT)
@app.route('/')
def home():
    return render_template("index.html", result=None)

# ✅ UPLOAD + TRAIN
@app.route('/upload', methods=['POST'])
def upload():
    global model
    try:
        crop_file = request.files['crop_file']
        weather_file = request.files['weather_file']

        # 🔹 SAFE FILE READING
        def read_file(file):
            try:
                return pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
            except:
                return pd.read_excel(file)

        crop = read_file(crop_file)
        weather = read_file(weather_file)

        # 🔹 CLEAN COLUMN NAMES
        crop.columns = crop.columns.str.strip()
        weather.columns = weather.columns.str.strip()

        # 🔹 VALIDATION
        if 'Year' not in crop.columns or 'Yield' not in crop.columns:
            return render_template("index.html", result="❌ Crop file must contain Year & Yield")

        if not all(col in weather.columns for col in ['Year', 'Rainfall', 'Temperature', 'Humidity']):
            return render_template("index.html", result="❌ Weather file missing required columns")

        # 🔹 MERGE
        data = pd.merge(crop, weather, on="Year")

        # 🔹 CLEAN DATA
        data = data.dropna()

        # Convert to numeric (important fix)
        data[['Rainfall', 'Temperature', 'Humidity', 'Yield']] = \
            data[['Rainfall', 'Temperature', 'Humidity', 'Yield']].apply(pd.to_numeric, errors='coerce')

        data = data.dropna()

        # 🔹 FEATURES
        X = data[['Rainfall', 'Temperature', 'Humidity']]
        y = data['Yield']

        # 🔹 TRAIN MODEL
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        # 🔹 SAVE MODEL
        pickle.dump(model, open(model_path, "wb"))

        return render_template("index.html", result="✅ Model updated successfully!")

    except Exception as e:
        return render_template("index.html", result=f"❌ Error: {str(e)}")

# ✅ FAST PREDICTION
@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return render_template("index.html", result="⚠️ Please upload data first")

        rainfall = float(request.form['rainfall'])
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])

        input_data = np.array([[rainfall, temp, humidity]])

        prediction = model.predict(input_data)

        return render_template("index.html", result=f"🌾 {round(prediction[0], 2)} tons/hectare")

    except Exception as e:
        return render_template("index.html", result=f"❌ Error: {str(e)}")

# RUN
if __name__ == "__main__":
    app.run(debug=True)