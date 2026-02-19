
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the saved model and scaler
model = joblib.load('final_random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. Define selected features (must be in the exact order as used during training)
selected_features = [
    'age_last_milestone_year',
    'relationships',
    'age_first_milestone_year',
    'funding_total_usd',
    'age_last_funding_year',
    'age_first_funding_year',
    'milestones',
    'time_to_first_funding',
    'avg_participants',
    'is_top500',
    'latitude',
    'funding_duration',
    'longitude',
    'funding_rounds',
    'has_roundC'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {}
    for feature in selected_features:
        input_data[feature] = float(request.form[feature])

    # Convert input data to DataFrame, ensuring correct order and shape
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
