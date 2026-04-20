from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle
import pandas as pd
import logging

# Hide all Flask/Werkzeug messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# Load best model
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
    features = pd.DataFrame([{f: data[f] for f in FIELDS}])

    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]  # [prob_class0, prob_class1]

    is_complete = int(prediction) == 1

    # Use the probability of the PREDICTED class as confidence
    # class 0 = Incomplete, class 1 = Complete
    confidence  = probabilities[1] if is_complete else probabilities[0]
    error       = 1.0 - confidence

    return jsonify({
        "result":     "Complete" if is_complete else "Incomplete",
        "confidence": round(float(confidence) * 100, 1),
        "error":      round(float(error) * 100, 1)
    })

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    app.run(port=5000)
