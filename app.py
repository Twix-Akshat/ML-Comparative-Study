import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

DATASET_PATH = os.path.join(os.getcwd(), "datasets", "housing.csv")

trained_models = {}  # Store trained models globally
scaler = None
feature_order = []  # Store the correct order of features

@app.route("/train_models", methods=["POST"])
def train_models():
    global trained_models, scaler, feature_order

    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Housing dataset not found! Ensure 'housing.csv' is in the 'datasets' folder."}), 400

    df = pd.read_csv(DATASET_PATH)

    binary_mapping = {"yes": 1, "no": 0}
    furnishing_mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

    for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        if col in df.columns:
            df[col] = df[col].replace(binary_mapping)

    if "furnishingstatus" in df.columns:
        df["furnishingstatus"] = df["furnishingstatus"].replace(furnishing_mapping)

    df.fillna(0, inplace=True)

    if df.isnull().sum().sum() > 0:
        return jsonify({"error": "Dataset contains missing values. Please clean your dataset."}), 400

    feature_order = list(df.columns[:-1])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf'),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = 1 - (mean_squared_error(y_test, y_pred) / y_test.var())

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
            "Accuracy": round(accuracy * 100, 2)
        }

        trained_models[name] = model

    return jsonify(results)

@app.route("/predict", methods=["POST"])
def predict():
    global trained_models, scaler, feature_order

    try:
        data = request.json
        features = data.get("features", {})
        model_name = data.get("model", "Linear Regression")

        if not features or model_name not in trained_models:
            return jsonify({"error": "Invalid input or model not trained yet"}), 400

        binary_mapping = {"yes": 1, "no": 0}
        furnishing_mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

        converted_features = []
        for key in feature_order:
            val = features.get(key, 0)
            if key in binary_mapping:
                converted_features.append(binary_mapping.get(val, 0))
            elif key == "furnishingstatus":
                converted_features.append(furnishing_mapping.get(val, 0))
            else:
                converted_features.append(float(val))

        X_input = pd.DataFrame([converted_features])
        X_input_scaled = scaler.transform(X_input)

        model = trained_models[model_name]
        prediction = model.predict(X_input_scaled)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=90000)
