import os
import re
import base64
import pickle
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI, RateLimitError
import pickle
import logging

# ── App & Config ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("TYPHOON_API_KEY", "")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_thai.pkl")

client = OpenAI(api_key=API_KEY, base_url="https://api.opentyphoon.ai/v1")

# Load trained model once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Constants ─────────────────────────────────────────────────────────────────
FIELDS = [
    "has_store_name",
    "has_store_address",
    "has_store_tax_id",
    "has_store_phone",
    "has_date",
    "has_invoice_number",
    "has_item_list",
    "has_amount",
    "has_vat",
    "has_amount_words",
    "has_signature",
]

FIELD_LABELS_TH = {
    "has_store_name":     "ชื่อร้าน/บริษัท",
    "has_store_address":  "ที่อยู่ร้าน",
    "has_store_tax_id":   "เลขประจำตัวผู้เสียภาษี",
    "has_store_phone":    "เบอร์โทรศัพท์",
    "has_date":           "วันที่",
    "has_invoice_number": "เลขที่ใบเสร็จ/ใบแจ้งหนี้",
    "has_item_list":      "รายการสินค้า",
    "has_amount":         "ยอดรวม",
    "has_vat":            "ภาษีมูลค่าเพิ่ม",
    "has_amount_words":   "จำนวนเงินเป็นตัวอักษร",
    "has_signature":      "ลายเซ็น/ผู้รับรอง",
}

THAI_MONTHS = (
    r"มกราคม|ม\.ค\.|กุมภาพันธ์|ก\.พ\.|มีนาคม|มี\.ค\.|เมษายน|เม\.ย\.|"
    r"พฤษภาคม|พ\.ค\.|มิถุนายน|มิ\.ย\.|กรกฎาคม|ก\.ค\.|"
    r"สิงหาคม|ส\.ค\.|กันยายน|ก\.ย\.|ตุลาคม|ต\.ค\.|"
    r"พฤศจิกายน|พ\.ย\.|ธันวาคม|ธ\.ค\."
)
ENG_MONTHS = (
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december"
)


# ── Helper ────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ── OCR ───────────────────────────────────────────────────────────────────────
def extract_text(image_bytes: bytes, mime: str) -> str:
    """Call Typhoon OCR with base64-encoded image bytes."""
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="typhoon-ocr",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                        {"type": "text",
                         "text": (
                             "กรุณาดึงข้อความทั้งหมดจากภาพใบเสร็จ/ใบแจ้งหนี้นี้ "
                             "รวมถึงภาษาไทยและภาษาอังกฤษ ทุกบรรทัด"
                         )},
                    ],
                }],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 60 * (attempt + 1)
            app.logger.warning(f"Rate limit — waiting {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
        except Exception as e:
            app.logger.error(f"OCR error: {e}")
            time.sleep(10)
    return ""


# ── Detection Functions (Thai-aware) ─────────────────────────────────────────
def detect_store_name(text: str) -> int:
    t = normalize(text)
    thai = ["จำกัด", "มหาชน", "บริษัท", "ห้างหุ้นส่วน", "ร้าน", "หจก.", "บจก.", "บมจ."]
    eng  = ["co.,ltd", "co., ltd", "ltd.", "company", "shop", "store", "bakery",
             "cafe", "restaurant", "hotel", "hospital", "clinic", "center", "centre",
             "service", "technology", "university", "inc.", "inc,"]
    return int(any(kw in t for kw in thai + eng))


def detect_store_address(text: str) -> int:
    thai_addr = [
        "หมู่", "ตำบล", "ต.", "อำเภอ", "อ.", "จังหวัด", "จ.",
        "ถนน", "ถ.", "ซอย", "ซ.", "แขวง", "เขต",
        "กรุงเทพ", "กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี",
        "เชียงราย", "เชียงใหม่", "ขอนแก่น", "โทร", "โทรศัพท์",
    ]
    eng_addr = ["road", "rd.", "street", "st.", "anywhere st", "moo", "any city"]
    for line in text.split("\n"):
        l = line.strip()
        if not l or len(l) < 4:
            continue
        if any(kw in l for kw in thai_addr):
            return 1
        if any(kw in l.lower() for kw in eng_addr):
            return 1
        if re.search(r"^\d+[\/]\d+", l):
            return 1
        if re.search(r"\b1\d{4}\b|\b[2-9]\d{4}\b", l):
            return 1
    return 0


def detect_store_tax_id(text: str) -> int:
    thai_labels = [
        "เลขประจำตัวผู้เสียภาษี", "เลขผู้เสียภาษี",
        "เลขประจำตัวนิติบุคคล", "tax id", "tax no",
    ]
    t_lower = text.lower()
    for label in thai_labels:
        if label in text or label in t_lower:
            idx = text.find(label) if label in text else t_lower.find(label)
            if re.search(r"\d{10,13}", text[idx: idx + 40]):
                return 1
    if re.findall(r"\b\d{13}\b", text):
        return 1
    if re.search(r"\b\d{4}[\-\s]?\d{3}[\-\s]?\d{5}[\-\s]?\d{1}\b", text):
        return 1
    return 0


def detect_store_phone(text: str) -> int:
    t = normalize(text)
    if any(kw in t for kw in ["โทร", "โทรศัพท์", "tel", "phone", "mobile"]):
        if re.search(r"\d{2,3}[\-\s]?\d{3,4}[\-\s]?\d{3,4}", text):
            return 1
    if re.search(r"\b0[6-9]\d[\-\s]?\d{3,4}[\-\s]?\d{4}\b", text):
        return 1
    if re.search(r"\b0\d[\-\s]\d{3,4}[\-\s]\d{4}\b", text):
        return 1
    if re.search(r"\+66[\-\s]?\d", text):
        return 1
    if re.search(r"\b\d{9,10}\b", text):
        return 1
    return 0


def detect_date(text: str) -> int:
    if re.search(r"วันที่\s*\d", text):
        return 1
    if re.search(THAI_MONTHS, text):
        return 1
    if re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b", text):
        return 1
    if re.search(r"(date|วัน)[:\s]+.{1,20}\d{2,4}", text, re.IGNORECASE):
        return 1
    if re.search(r"\b\d{1,2}\s+(" + ENG_MONTHS + r")\s+\d{4}\b", text, re.IGNORECASE):
        return 1
    return 0


def detect_invoice_number(text: str) -> int:
    t_lower = text.lower()
    thai_labels = [
        "เลขที่", "ใบแจ้งหนี้เลขที่", "เลขที่ใบแจ้งหนี้",
        "เลขที่ใบเสร็จ", "เลขที่เอกสาร", "หมายเลข",
    ]
    for label in thai_labels:
        if label in text:
            idx = text.find(label)
            if re.search(r"[A-Z\d][\w\-\/]{2,}", text[idx: idx + 30]):
                return 1
    if re.search(r"(invoice|receipt|inv|no\.?|#)[\s:]+[A-Z\d][\w\-\/]{2,}", t_lower):
        return 1
    if re.search(r"\b(?:IV|SR|DS|BK|IVEN|lv)\d+", text, re.IGNORECASE):
        return 1
    if re.search(r"\b\d{4,6}[\-\/]\d{3,}\b", text):
        return 1
    return 0


def detect_item_list(text: str) -> int:
    t = normalize(text)
    thai_headers = ["รายการ", "รายละเอียด", "รายการสินค้า", "สินค้า",
                    "รายละเอียดสินค้า", "ลำดับ"]
    if any(kw in t for kw in thai_headers):
        return 1
    if any(kw in t for kw in ["description", "item", "qty", "quantity", "unit price"]):
        return 1
    if len(re.findall(r"[฿$]?[\d,]+\.\d{2}", text)) >= 2:
        return 1
    return 0


def detect_amount(text: str) -> int:
    thai_total = [
        "ยอดรวมทั้งหมด", "จำนวนเงินรวมทั้งสิ้น", "รวมทั้งสิ้น",
        "ยอดรวม", "รวมเป็นเงิน", "ราคารวมสุทธิ", "ยอดชำระ",
        "รวมสุทธิ", "จำนวนเงินทั้งสิ้น", "รวมทั้งหมด",
    ]
    for kw in thai_total:
        if kw in text:
            idx = text.find(kw)
            if re.search(r"[฿]?[\d,]{3,}", text[idx: idx + 60]):
                return 1
    if re.search(r"(grand\s*total|total\s*amount)[\s:]*[฿$]?[\d,]+\.?\d*",
                 text, re.IGNORECASE):
        return 1
    return 0


def detect_vat(text: str) -> int:
    t = normalize(text)
    if any(kw in t for kw in ["ภาษีมูลค่าเพิ่ม", "ภาษี", "vat"]):
        return 1
    if re.search(r"vat\s*\(?\d+%?\)?", t):
        return 1
    return 0


def detect_amount_in_words(text: str) -> int:
    t = normalize(text)
    if "บาทถ้วน" in text or "บาทถ้วน" in t:
        return 1
    thai_num_words = ["หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก",
                      "เจ็ด", "แปด", "เก้า", "สิบ", "ร้อย", "พัน",
                      "หมื่น", "แสน", "ล้าน"]
    if any(w in text for w in thai_num_words) and "บาท" in text:
        for w in thai_num_words:
            if w in text:
                idx = text.find(w)
                if "บาท" in text[idx: idx + 60]:
                    return 1
    if re.search(r"baht\s+only", t):
        return 1
    return 0


def detect_signature(text: str) -> int:
    t = normalize(text)
    keywords = [
        "ผู้รับเงิน", "ผู้จ่ายเงิน", "ผู้อนุมัติ", "ผู้ตรวจสอบ",
        "ผู้จัดการ", "ผู้รับใบแจ้งหนี้", "ผู้ออกใบแจ้งหนี้",
        "ผู้รับบิล", "ผู้วางบิล", "เจ้าหน้าที่การเงิน",
        "ผู้รับสินค้า", "ผู้จัดทำ", "ผู้ส่งสินค้า",
        "cashier", "signature", "signed by", "approved by",
        "received by", "authorised by", "delivery by",
    ]
    return int(any(kw in t for kw in keywords))


DETECTORS = {
    "has_store_name":     detect_store_name,
    "has_store_address":  detect_store_address,
    "has_store_tax_id":   detect_store_tax_id,
    "has_store_phone":    detect_store_phone,
    "has_date":           detect_date,
    "has_invoice_number": detect_invoice_number,
    "has_item_list":      detect_item_list,
    "has_amount":         detect_amount,
    "has_vat":            detect_vat,
    "has_amount_words":   detect_amount_in_words,
    "has_signature":      detect_signature,
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png"):
        return jsonify({"error": "Only JPG/PNG images accepted"}), 400

    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    image_bytes = file.read()

    # Step 1: OCR
    ocr_text = extract_text(image_bytes, mime)
    if not ocr_text:
        return jsonify({"error": "OCR failed — check API key or try again"}), 500

    # Step 2: Run detectors
    field_results = {field: DETECTORS[field](ocr_text) for field in FIELDS}

    # Step 3: ML prediction
    feature_vector = [[field_results[f] for f in FIELDS]]
    prediction = int(model.predict(feature_vector)[0])
    proba = model.predict_proba(feature_vector)[0].tolist()
    confidence = round(max(proba) * 100, 1)

    # Build per-field detail list for UI
    fields_detail = [
        {
            "key":    field,
            "label":  FIELD_LABELS_TH[field],
            "found":  bool(field_results[field]),
        }
        for field in FIELDS
    ]

    found_count = sum(field_results[f] for f in FIELDS)

    return jsonify({
        "prediction":   prediction,
        "label_text":   "ใบเสร็จสมบูรณ์ ✓" if prediction == 1 else "ใบเสร็จไม่สมบูรณ์ ✗",
        "confidence":   confidence,
        "found_count":  found_count,
        "total_fields": len(FIELDS),
        "fields":       fields_detail,
        "ocr_text":     ocr_text,
    })


if __name__ == "__main__":
    if not API_KEY:
        print("⚠️  Warning: TYPHOON_API_KEY not set. OCR will fail.")
    if not os.path.exists(MODEL_PATH):
        print("⚠️  Warning: best_model_thai.pkl not found. Run the notebook first.")
    print("🚀 Starting Thai Receipt Validation Server at http://localhost:5000")
    app.run(debug=True, port=5000)
