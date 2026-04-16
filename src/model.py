import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ✅ BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "real_final_dataset.csv")
model_path = os.path.join(BASE_DIR, "models", "yield_model.pkl")

# ✅ LOAD DATA
data = pd.read_csv(data_path)

print("Dataset shape:", data.shape)

# ✅ CLEAN DATA
data = data.dropna()
data = data.drop_duplicates()

# Convert to numeric (safe)
data["Temperature"] = pd.to_numeric(data["Temperature"], errors="coerce")
data["Rainfall"] = pd.to_numeric(data["Rainfall"], errors="coerce")
data["Humidity"] = pd.to_numeric(data["Humidity"], errors="coerce")
data["Yield"] = pd.to_numeric(data["Yield"], errors="coerce")

data = data.dropna()

# Encode categorical features
data = pd.get_dummies(data, columns=["Crop", "District"], drop_first=True)

# Use all features except target
X = data.drop(columns=["Yield"])
y = data["Yield"]

# ✅ SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ MODEL
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=2,
    scoring="r2",
    n_jobs=-1
)

# ✅ TRAIN
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ✅ EVALUATE
pred = best_model.predict(X_test)
score = r2_score(y_test, pred)

print("✅ R2 Score:", score)

# ✅ SAVE MODEL
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print("🔥 Model saved successfully at:", model_path)