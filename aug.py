import os
import pandas as pd

augmented_folder = "augmented_thaireceipts"
output_csv = "augmented_labels.csv"

rows = []

for file in os.listdir(augmented_folder):
    if file.endswith((".jpg", ".png")):
        rows.append({
            "filename": file,
            "has_store_name": 0,
            "has_store_address": 0,
            "has_store_tax_id": 0,
            "has_date": 0,
            "has_amount": 0,
            "has_amount_words": 0,
            "has_signature": 0,
            "has_mfu_name": 0,
            "has_mfu_address": 0,
            "has_mfu_tax_id": 0,
            "label": 0
        })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print("✅ Empty label CSV created")