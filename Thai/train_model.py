import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Training data ─────────────────────────────────────────────
# Each row: [mfu_name, mfu_addr, mfu_tax, store_name, store_addr,
#            store_tax, date, amount, amount_words, signature, label]
data = [
    # Complete receipts (label=1)
    [1,1,1,1,1,1,1,1,1,1, 1],
    [1,1,1,1,1,1,1,1,1,1, 1],
    [1,1,1,1,1,1,1,1,0,1, 1],
    [1,1,1,1,1,1,1,1,1,0, 1],
    [1,1,1,1,0,1,1,1,1,1, 1],
    [1,1,1,1,1,0,1,1,1,1, 1],
    [1,1,1,1,1,1,1,1,1,1, 1],
    [1,1,1,1,1,1,1,1,1,1, 1],

    # Incomplete receipts (label=0)
    [0,0,0,1,1,1,1,1,1,1, 0],
    [1,1,1,0,0,0,1,1,1,1, 0],
    [1,1,1,1,1,1,0,0,0,0, 0],
    [0,1,1,1,1,1,1,0,0,1, 0],
    [1,0,0,0,0,0,1,1,1,1, 0],
    [1,1,0,1,1,0,0,1,0,0, 0],
    [0,0,0,0,0,0,0,0,0,0, 0],
    [1,1,1,1,1,1,1,0,0,0, 0],
]

data = np.array(data)
X = data[:, :10]
y = data[:, 10]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

with open("best_model_thai.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Saved: best_model_thai.pkl")