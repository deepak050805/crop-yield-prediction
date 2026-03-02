from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ✅ Fix path (important for your structure)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "yield_model.pkl")

# ✅ Load model ONCE
model = pickle.load(open(model_path, "rb"))

# HOME
@app.route('/')
def home():
    return render_template("index.html", result=None)

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    try:
        rainfall = float(request.form['rainfall'])
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])

        # ✅ Fast numpy input
        input_data = np.array([[rainfall, temp, humidity]])

        prediction = model.predict(input_data)

        return render_template("index.html", result=round(prediction[0], 2))

    except Exception as e:
        return f"Error: {str(e)}"

# RUN
if __name__ == "__main__":
    app.run(debug=True)