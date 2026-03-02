import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

crop = pd.read_csv("data/crop_data.csv")
weather = pd.read_csv("data/weather_data.csv")

data = pd.merge(crop, weather, on="Year")

X = data[['Rainfall', 'Temperature', 'Humidity']]
y = data['Yield']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("models/yield_model.pkl", "wb"))

print("Model saved")