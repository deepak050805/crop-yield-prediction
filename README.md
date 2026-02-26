# 🌾 Crop Yield Predictor

A machine learning-based web application that predicts crop yield using key weather parameters like rainfall, temperature, and humidity. This project combines data analysis, machine learning, and a web interface to provide real-time predictions.

---

## 🚀 Project Overview

Agriculture is highly dependent on environmental conditions. This project helps in estimating crop yield based on historical weather data, allowing better planning and decision-making.

The model is trained using real datasets and deployed through a Flask web application where users can input weather conditions and get predictions instantly.

---

## 🎯 Features

- 🌧 Input weather parameters (Rainfall, Temperature, Humidity)
- 🤖 Predict crop yield using Machine Learning
- 🌐 Interactive web interface using Flask
- 📊 Data analysis using Pandas and visualization tools
- 💾 Model saved using Pickle for reuse

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn (Linear Regression)  
- Flask  
- HTML & CSS  

---

## 📂 Project Structure
crop-yield-project/
│
├── app/
│ ├── app.py
│ └── templates/
│ └── index.html
│
├── data/
│ ├── crop_data.csv
│ └── weather_data.csv
│
├── models/
│ └── yield_model.pkl
│
├── notebooks/
│ └── analysis.ipynb
│
├── src/
│ └── model.py
│
└── README.md
