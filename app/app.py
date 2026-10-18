import json
import math
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE_DIR / "data" / "real_final_dataset.csv")
districts = sorted(df["District"].unique().tolist())
crops = sorted(df["Crop"].unique().tolist())
with (BASE_DIR / "models" / "yield_model.pkl").open("rb") as stream:
    model = pickle.load(stream)


def prepare_input(temp, rainfall, humidity, crop, district):
    return pd.DataFrame([{
        "Temperature": temp, "Rainfall": rainfall, "Humidity": humidity,
        "Crop": crop, "District": district,
    }])


def page(template, active):
    return render_template(template, active=active, districts=districts, crops=crops,
                           first_year=int(df["Year"].min()), last_year=int(df["Year"].max()))


@app.get("/")
def home():
    return page("index.html", "overview")


@app.get("/predictor")
def predictor():
    return page("predictor.html", "predictor")


@app.get("/history")
def history():
    return page("history.html", "history")


@app.post("/get_data")
def get_data():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object with crop and district."), 400
    district, crop = data.get("district"), data.get("crop")
    if district not in districts or crop not in crops:
        return jsonify(error="Choose a valid crop and district."), 400
    filtered = df[(df["District"] == district) & (df["Crop"] == crop)]
    return jsonify(filtered.sort_values("Year").to_dict(orient="records"))


@app.get("/model_info")
def model_info():
    metrics_path = BASE_DIR / "models" / "model_metrics.json"
    if not metrics_path.exists():
        return jsonify(error="Model evaluation is not available. Run python src/model.py."), 503
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))


@app.post("/predict")
def predict():
    try:
        crop, district = request.form.get("crop"), request.form.get("district")
        if crop not in crops or district not in districts:
            return jsonify(error="Choose a valid crop and district."), 400
        rainfall = float(request.form["rainfall"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        if not all(math.isfinite(value) for value in [rainfall, temp, humidity]):
            return jsonify(error="Weather inputs must be finite numbers."), 400
        if not 0 <= rainfall <= 2000 or not -20 <= temp <= 60 or not 0 <= humidity <= 100:
            return jsonify(error="Use rainfall from 0 to 2000 mm, temperature from -20 to 60 C, and humidity from 0 to 100%."), 400

        observed = df[df["Crop"] == crop]
        bounds = {column: (float(observed[column].min()), float(observed[column].max()))
                  for column in ["Rainfall", "Temperature", "Humidity"]}
        warnings = []
        for column, value in [("Rainfall", rainfall), ("Temperature", temp), ("Humidity", humidity)]:
            low, high = bounds[column]
            if not low <= value <= high:
                warnings.append(f"{column} is outside this crop's historical range ({low:g}-{high:g}); the estimate is less reliable.")

        options = [(rainfall, temp, humidity)]
        for direction in [1, -1]:
            values = []
            for column, value, step in [("Rainfall", rainfall, 100),
                                        ("Temperature", temp, 2), ("Humidity", humidity, 5)]:
                low, high = bounds[column]
                values.append(min(high, max(low, value + direction * step)))
            options.append(tuple(values))
        inputs = pd.concat([prepare_input(t, r, h, crop, district) for r, t, h in options], ignore_index=True)
        yields = model.predict(inputs)
        best = int(yields.argmax())
        best_rain, best_temp, best_hum = options[best]
        predicted, best_yield = float(yields[0]), float(yields[best])
        return jsonify(
            predicted=round(predicted, 3), optimal=round(best_yield, 3),
            gap=round(best_yield - predicted, 3), best_rain=round(best_rain, 1),
            best_temp=round(best_temp, 1), best_hum=round(best_hum, 1),
            crop=crop, district=district, warnings=warnings,
        )
    except (ValueError, KeyError, TypeError):
        return jsonify(error="Provide valid rainfall, temperature, and humidity values."), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction could not be completed. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
