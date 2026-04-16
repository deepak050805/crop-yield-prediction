from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ✅ BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "real_final_dataset.csv")
model_path = os.path.join(BASE_DIR, "models", "yield_model.pkl")

# ✅ LOAD DATA
df = pd.read_csv(data_path)

districts = sorted(df["District"].unique().tolist())
crops = sorted(df["Crop"].unique().tolist())

# ✅ LOAD MODEL
model = pickle.load(open(model_path, "rb"))

# ✅ GET MODEL COLUMNS (VERY IMPORTANT)
model_columns = model.feature_names_in_


# 🔥 PREPARE INPUT FUNCTION (MOST IMPORTANT PART)
def prepare_input(temp, rainfall, humidity, crop, district):
    input_dict = {col: 0 for col in model_columns}

    input_dict["Temperature"] = temp
    input_dict["Rainfall"] = rainfall
    input_dict["Humidity"] = humidity

    crop_col = f"Crop_{crop}"
    district_col = f"District_{district}"

    if crop_col in input_dict:
        input_dict[crop_col] = 1

    if district_col in input_dict:
        input_dict[district_col] = 1

    return pd.DataFrame([input_dict])


# ✅ HOME
@app.route("/")
def home():
    return render_template("index.html", districts=districts, crops=crops)


# ✅ GET DATA FOR CHARTS
@app.route("/get_data", methods=["POST"])
def get_data():
    data = request.get_json()

    district = data["district"]
    crop = data["crop"]

    filtered = df[
        (df["District"] == district) &
        (df["Crop"] == crop)
    ]

    return jsonify(filtered.to_dict(orient="records"))


# ✅ PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        rainfall = float(request.form.get("rainfall", 0))
        temp = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))

        crop = request.form.get("crop")
        district = request.form.get("district")

        # 🔥 USE PREPARE FUNCTION
        input_data = prepare_input(temp, rainfall, humidity, crop, district)

        predicted = float(model.predict(input_data)[0])

        # 🔥 TRY NEARBY CONDITIONS
        options = [
            (rainfall + 100, temp + 2, humidity + 5),
            (rainfall - 100, temp - 2, humidity - 5)
        ]

        best_yield = predicted
        best_rain, best_temp, best_hum = rainfall, temp, humidity

        for r, t, h in options:
            temp_input = prepare_input(t, r, h, crop, district)

            y = float(model.predict(temp_input)[0])

            if y > best_yield:
                best_yield = y
                best_rain = r
                best_temp = t
                best_hum = h

        gap = best_yield - predicted

        return jsonify({
            "predicted": round(predicted, 2),
            "optimal": round(best_yield, 2),
            "gap": round(gap, 2),
            "best_rain": round(best_rain, 1),
            "best_temp": round(best_temp, 1),
            "best_hum": round(best_hum, 1),
            "crop": crop,
            "district": district
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ✅ RUN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)