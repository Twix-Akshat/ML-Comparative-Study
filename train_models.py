import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load Housing Dataset
df = pd.read_csv("housing.csv")  # Update with actual dataset path

# Feature Selection
X = df.drop(columns=["price"])  # Assuming 'price' is the target column
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(kernel="rbf"),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train Models & Save Performance
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "MSE": mse, "R2 Score": r2}
    
    # Save Model
    with open(f"{name.replace(' ', '_').lower()}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save Results
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

print("✅ Models trained and saved successfully!")
