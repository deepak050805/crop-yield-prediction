import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# LOAD DATA
data = pd.read_csv("data/historical_data.csv")

print("Dataset shape:", data.shape)

# ENCODE
data = pd.get_dummies(data, columns=['District','Crop'], drop_first=True)

# FEATURES
X = data.drop(columns=['Yield'])
y = data['Yield']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=2,
    scoring="r2",
    n_jobs=-1
)

# TRAIN
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# EVALUATE
pred = best_model.predict(X_test)
score = r2_score(y_test, pred)

print("✅ R2 Score:", score)

# SAVE
pickle.dump(best_model, open("models/yield_model.pkl","wb"))

print("🔥 Model saved successfully")