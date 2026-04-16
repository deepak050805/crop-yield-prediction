from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ✅ BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "historical_data.csv")
model_path = os.path.join(BASE_DIR, "models", "yield_model.pkl")

df = pd.read_csv(data_path)

districts = sorted(df["District"].unique().tolist())
crops = sorted(df["Crop"].unique().tolist())

model = pickle.load(open(model_path, "rb"))

# ✅ FAST INPUT PREP FUNCTION (KEY OPTIMIZATION)
def prepare_input(r, t, h, crop, district):
    df_input = pd.DataFrame([{
        "Rainfall": r,
        "Temperature": t,
        "Humidity": h,
        "District": district,
        "Crop": crop
    }])

    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    return df_input


@app.route("/")
def home():
    return render_template("index.html", districts=districts, crops=crops)


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


@app.route("/predict", methods=["POST"])
def predict():
    try:
        rainfall = float(request.form.get("rainfall", 0))
        temp = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))

        crop = request.form.get("crop")
        district = request.form.get("district")

        # 🔹 SINGLE PREP (FAST)
        input_data = prepare_input(rainfall, temp, humidity, crop, district)
        predicted = float(model.predict(input_data)[0])

        # 🔥 REDUCED OPTIONS (FAST)
        options = [
            (rainfall + 100, temp + 2, humidity + 5),
            (rainfall - 100, temp - 2, humidity - 5)
        ]

        best_yield = predicted
        best_rain, best_temp, best_hum = rainfall, temp, humidity

        # 🔥 FAST LOOP (reusing function)
        for r, t, h in options:
            temp_input = prepare_input(r, t, h, crop, district)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)