from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
with open("best_model_thai.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_KEYS = [
    "has_mfu_name",
    "has_mfu_address", 
    "has_mfu_tax_id",
    "has_store_name",
    "has_store_address",
    "has_store_tax_id",
    "has_date",
    "has_amount",
    "has_amount_words",
    "has_signature"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[data.get(k, 0) for k in FEATURE_KEYS]])
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        confidence = round(float(max(proba)) * 100, 1)
        result = "Complete" if prediction == 1 else "Incomplete"
        
        return jsonify({
            "result": result,
            "prediction": int(prediction),
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)