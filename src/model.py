import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Load datasets
crop = pd.read_csv("data/crop_data.csv")
weather = pd.read_csv("data/weather_data.csv")

# Merge datasets
data = pd.merge(crop, weather, on="Year")

# Prepare data
X = data[['Rainfall', 'Temperature', 'Humidity']]
y = data['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model
pickle.dump(model, open("models/yield_model.pkl", "wb"))

print("Model saved successfully ")