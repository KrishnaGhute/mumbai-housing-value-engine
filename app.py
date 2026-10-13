
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mumbai_housing_rf_model.pkl"

# Retain the existing project's income-scaling convention.
REFERENCE_MONTHLY_INCOME = 30000
DATASET_MEDIAN_INCOME = 0.815520

MIN_MEDIAN_INCOME = 0.027715
MAX_MEDIAN_INCOME = 3.613535

# Supported geographic area for this application.
MIN_LAT, MAX_LAT = 18.80, 19.35
MIN_LNG, MAX_LNG = 72.70, 73.10

EXPECTED_FEATURES = [
    "median_income",
    "latitude",
    "longitude"
]

model = None
model_features = EXPECTED_FEATURES

try:
    model = joblib.load(MODEL_PATH)

    saved_features = list(model.feature_names_in_)

    if set(saved_features) != set(EXPECTED_FEATURES):
        raise ValueError(
            f"Unexpected model features: {saved_features}"
        )

    # Use the exact feature order from training.
    model_features = saved_features

    print("Model loaded successfully.")
    print("Model type:", type(model).__name__)
    print("Features:", model_features)

except Exception as exc:
    model = None
    print("MODEL LOAD ERROR:", exc)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "The prediction model could not be loaded."
        }), 500

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid input data."
            }), 400

        monthly_income = float(data.get("monthly_income", 0))
        latitude = float(data.get("latitude", 0))
        longitude = float(data.get("longitude", 0))

        values = [monthly_income, latitude, longitude]

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Enter valid numeric values.")

        if monthly_income <= 0:
            raise ValueError(
                "Monthly household income must be greater than zero."
            )

        if not MIN_LAT <= latitude <= MAX_LAT:
            raise ValueError(
                "Move the marker inside the supported Mumbai region."
            )

        if not MIN_LNG <= longitude <= MAX_LNG:
            raise ValueError(
                "Move the marker inside the supported Mumbai region."
            )

        # Same conversion convention used in your earlier app.
        median_income = (
            monthly_income / REFERENCE_MONTHLY_INCOME
        ) * DATASET_MEDIAN_INCOME

        median_income = float(np.clip(
            median_income,
            MIN_MEDIAN_INCOME,
            MAX_MEDIAN_INCOME
        ))

        # Only the three features used by the saved model.
        input_data = pd.DataFrame([{
            "median_income": median_income,
            "latitude": latitude,
            "longitude": longitude
        }])

        input_data = input_data[model_features]

        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, prediction)

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })

    except (ValueError, TypeError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Unable to generate a prediction right now."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
