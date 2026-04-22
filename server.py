from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging

# Hide Flask startup noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# ── Load both models ──────────────────────────────────────────────────────────
with open("best_model.pkl", "rb") as f:
    model_eng = pickle.load(f)

with open("best_model_thai.pkl", "rb") as f:
    model_thai = pickle.load(f)

# ── Feature keys for each model ───────────────────────────────────────────────
FIELDS_ENG = [
    "has_store_name", "has_store_address", "has_store_tax_id",
    "has_date", "has_amount", "has_amount_words",
    "has_signature", "has_mfu_name", "has_mfu_address", "has_mfu_tax_id"
]

FIELDS_THAI = [
    "has_mfu_name", "has_mfu_address", "has_mfu_tax_id",
    "has_store_name", "has_store_address", "has_store_tax_id",
    "has_date", "has_amount", "has_amount_words", "has_signature"
]

# ── Helper: get model info string ─────────────────────────────────────────────
def get_model_name(model):
    """Return a short human-readable model name."""
    cls = type(model).__name__
    name_map = {
        "GradientBoostingClassifier": "Gradient Boosting",
        "RandomForestClassifier":     "Random Forest",
        "LogisticRegression":         "Logistic Regression",
        "SVC":                        "Support Vector Machine",
        "DecisionTreeClassifier":     "Decision Tree",
        "AdaBoostClassifier":         "AdaBoost",
        "XGBClassifier":              "XGBoost",
        "LGBMClassifier":             "LightGBM",
    }
    return name_map.get(cls, cls)


# ── English Receipt endpoint ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_eng():
    try:
        data = request.get_json()
        features = np.array([[data.get(k, 0) for k in FIELDS_ENG]])

        prediction  = model_eng.predict(features)[0]
        proba       = model_eng.predict_proba(features)[0]
        is_complete = int(prediction) == 1
        confidence  = round(float(proba[1] if is_complete else proba[0]) * 100, 1)
        error       = round(100 - confidence, 1)

        return jsonify({
            "result":     "Complete" if is_complete else "Incomplete",
            "confidence": confidence,
            "error":      error,
            "model_name": get_model_name(model_eng)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Thai Receipt endpoint ──────────────────────────────────────────────────────
@app.route("/predict_thai", methods=["POST"])
def predict_thai():
    try:
        data = request.get_json()
        features = np.array([[data.get(k, 0) for k in FIELDS_THAI]])

        prediction  = model_thai.predict(features)[0]
        proba       = model_thai.predict_proba(features)[0]
        is_complete = int(prediction) == 1
        confidence  = round(float(proba[1] if is_complete else proba[0]) * 100, 1)
        error       = round(100 - confidence, 1)

        return jsonify({
            "result":     "Complete" if is_complete else "Incomplete",
            "confidence": confidence,
            "error":      error,
            "model_name": get_model_name(model_thai)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  Unified Receipt Validation Server")
    print("  English  → POST http://127.0.0.1:5000/predict")
    print("  Thai     → POST http://127.0.0.1:5000/predict_thai")
    print("=" * 55)
    print(f"  ENG model : {get_model_name(model_eng)}")
    print(f"  THAI model: {get_model_name(model_thai)}")
    print("  Press CTRL+C to quit")
    app.run(port=5000)
