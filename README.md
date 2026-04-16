# 🌾 Crop Yield Prediction & Optimization System

An intelligent machine learning-based web application that predicts crop yield and suggests optimal weather conditions to maximize agricultural productivity.

---

## 🚀 Overview

This project combines **machine learning + data visualization** to help farmers and researchers:

* Predict crop yield based on weather conditions
* Identify optimal environmental conditions
* Compare current vs best possible yield
* Get actionable insights for improving production

---

## 🧠 Key Features

### 📊 1. Yield Prediction

* Predict crop yield (tons/hectare)
* Based on:

  * Rainfall
  * Temperature
  * Humidity
  * Crop type
  * District

---

### 🌦️ 2. Optimal Weather Detection

* Finds best weather conditions using model-based optimization
* Suggests:

  * Ideal rainfall
  * Ideal temperature
  * Ideal humidity

---

### 📈 3. Visual Dashboard

* Yield trend over years
* Rainfall vs yield scatter plot
* Predicted vs optimal yield comparison graph

---

### ⚡ 4. Smart Optimization Engine

* Instead of static data, system:

  * Tests multiple weather combinations
  * Selects conditions with highest predicted yield

---

### 💡 5. Insight Generation

* Shows difference between current and optimal yield
* Helps users understand improvement potential

---

## 🏗️ Tech Stack

### 🔹 Backend

* Python
* Flask
* Pandas
* Scikit-learn

### 🔹 Frontend

* HTML
* CSS
* JavaScript
* Chart.js

---

## 🤖 Machine Learning

### Model Used:

* Random Forest Regressor

### Input Features:

* Rainfall
* Temperature
* Humidity
* Crop
* District

### Output:

* Crop Yield (t/ha)

---

## ⚙️ How It Works

1. User inputs weather conditions
2. Model predicts yield
3. System generates multiple variations of weather
4. Best combination is selected
5. Optimal yield is calculated
6. Results are visualized

---

## 📊 Example

| Parameter       | Value    |
| --------------- | -------- |
| Crop            | Wheat    |
| District        | Amritsar |
| Predicted Yield | 3.0 t/ha |
| Optimal Yield   | 3.2 t/ha |

👉 Insight: Yield can improve by ~6.6%

---

## 📂 Project Structure

```
Crop Yield Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── historical_data.csv
│
├── models/
│   └── yield_model.pkl
│
├── src/
│   └── model.py
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## ▶️ Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Flask app

```
python app/app.py
```

### 3. Open in browser

```
http://127.0.0.1:10000
```

---

## 🔥 Future Enhancements

* Smart recommendation system (actionable suggestions)
* Real-time weather API integration
* XGBoost model for higher accuracy
* Crop recommendation system
* PDF report generation

---

## 🧠 Key Learnings

* Regression modeling for real-world problems
* Feature engineering & preprocessing
* Model optimization techniques
* Data visualization using Chart.js
* Building full-stack ML applications

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to support **data-driven agricultural decisions**, improving yield and optimizing environmental conditions.




