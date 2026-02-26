import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 🔹 AUTO LOAD FILES
files = os.listdir("data")

crop_file = None
weather_file = None

for file in files:
    if "crop" in file.lower():
        crop_file = file
    elif "weather" in file.lower():
        weather_file = file

# 🔹 LOAD DATA
crop = pd.read_csv(f"data/{crop_file}")
weather = pd.read_csv(f"data/{weather_file}")

# 🔹 MERGE DATA
data = pd.merge(crop, weather, on="Year")

# 🔹 FEATURES
X = data[['Rainfall', 'Temperature', 'Humidity']]
y = data['Yield']

# 🔹 TRAIN MODEL
model = LinearRegression()
model.fit(X, y)

# 🔹 SAVE MODEL
pickle.dump(model, open("models/yield_model.pkl", "wb"))

print("Model trained using:", crop_file, "and", weather_file)