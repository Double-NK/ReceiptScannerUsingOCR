from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # allows HTML to call this server

# Load your best model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

FIELDS = [
    "has_store_name", "has_store_address", "has_store_tax_id",
    "has_date", "has_amount", "has_amount_words",
    "has_signature", "has_mfu_name", "has_mfu_address", "has_mfu_tax_id"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [[data[f] for f in FIELDS]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return jsonify({
        "result": "Complete" if prediction == 1 else "Incomplete",
        "confidence": round(float(probability) * 100, 1)
    })

if __name__ == "__main__":
    app.run(port=5000)