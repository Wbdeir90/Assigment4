from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load IRIS dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the IRIS Dataset API!",
        "routes": {
            "/data": "Fetch the dataset",
            "/predict": "Make a prediction (POST with JSON {'features': [...]})"
        }
    })

@app.route('/data', methods=['GET'])
def get_data():
    try:
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Error fetching dataset: {e}")
        return jsonify({"error": "Failed to load dataset"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({"prediction": int(prediction[0])})

    except ValueError as ve:
        logging.error(f"Invalid input: {ve}")
        return jsonify({"error": "Invalid input format"}), 400
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
