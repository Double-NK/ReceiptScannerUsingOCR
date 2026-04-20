"""
Receipt Classification & Signature Detection System
Complete pipeline in one file - Simple and Easy!
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import smart_resize

from config import *


# ============================================================================
# PART 1: TYPHOON OCR INTEGRATION
# ============================================================================

def extract_text_typhoon_ocr(image_path):
    """Extract text from image using Typhoon OCR API"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            headers = {'Authorization': f'Bearer {TYPHOON_API_KEY}'}
            response = requests.post('https://api.typhoon-ocr.com/v1/ocr', files=files, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            return ""
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


# ============================================================================
# PART 2: FIELD DETECTION & FEATURE EXTRACTION
# ============================================================================

MONTHS = "jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december"

def detect_field(text, keywords):
    """Detect if text contains keywords"""
    return any(k in text.lower() for k in keywords)

def extract_features(text):
    """Extract 7 features from OCR text"""
    t = text.lower()
    
    has_date = int(bool(re.search(rf"({MONTHS})\s+\d{{1,2}}", t)))
    has_name = int(detect_field(t, ['company', 'ltd', 'shop', 'restaurant', 'name', 'ชื่อ']))
    has_item = int(detect_field(t, ['description', 'qty', 'amount', 'total', 'price', 'บาท']))
    has_sig = int(detect_field(t, ['signature', 'received', 'ลายเซ็น', 'sign']))
    has_addr = int(detect_field(t, ['address', 'mae fah luang', 'mfu']))
    text_len = len(t) / 100  # Normalize
    digit_cnt = sum(1 for c in t if c.isdigit()) / 10  # Normalize
    
    return [has_date, has_name, has_item, has_sig, has_addr, text_len, digit_cnt]


# ============================================================================
# PART 3: RECEIPT CLASSIFICATION
# ============================================================================

class ReceiptClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train classifier"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("✓ Receipt classifier trained")
    
    def evaluate(self, X_test, y_test):
        """Evaluate and return metrics"""
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }
    
    def predict(self, features):
        """Predict single receipt"""
        features = np.array(features).reshape(1, -1)
        return self.model.predict(self.scaler.transform(features))[0]


# ============================================================================
# PART 4: SIGNATURE DETECTION - CNN
# ============================================================================

def build_cnn_model():
    """Build CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# PART 5: TRANSFER LEARNING - ResNet50
# ============================================================================

def build_transfer_learning_model():
    """Build Transfer Learning model with ResNet50"""
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# PART 6: MAIN PIPELINE
# ============================================================================

def load_images(folder_path, target_size):
    """Load images from folder"""
    images = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder_path, filename)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img.astype('float32') / 255.0
                    images.append(img)
    return np.array(images)

def run_pipeline():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("RECEIPT CLASSIFICATION & SIGNATURE DETECTION PIPELINE")
    print("="*70)
    
    # ========== PHASE 1: RECEIPT CLASSIFICATION ==========
    print("\n[PHASE 1] Receipt Classification")
    print("-" * 70)
    
    # Extract features from receipts
    features_list = []
    labels_list = []
    
    print("Extracting text from receipts...")
    
    # Complete receipts
    for filename in os.listdir(COMPLETE_PATH)[:5]:  # First 5 files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(COMPLETE_PATH, filename)
            text = extract_text_typhoon_ocr(path)
            if text:
                features = extract_features(text)
                features_list.append(features)
                labels_list.append(1)
                print(f"  ✓ {filename} (Complete)")
    
    # Incomplete receipts
    for filename in os.listdir(INCOMPLETE_PATH)[:5]:  # First 5 files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(INCOMPLETE_PATH, filename)
            text = extract_text_typhoon_ocr(path)
            if text:
                features = extract_features(text)
                features_list.append(features)
                labels_list.append(0)
                print(f"  ✓ {filename} (Incomplete)")
    
    if len(features_list) < 4:
        print("\n⚠ Limited data - creating synthetic dataset for demo...")
        features_list = [
            [1, 1, 1, 1, 1, 5, 15],  # Complete
            [1, 1, 1, 0, 1, 4, 10],  # Incomplete
            [1, 0, 1, 1, 0, 3, 8],   # Incomplete
            [0, 1, 1, 1, 1, 6, 20],  # Complete
        ]
        labels_list = [1, 0, 0, 1]
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    
    # Train classifier
    classifier = ReceiptClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate
    receipt_metrics = classifier.evaluate(X_test, y_test)
    
    print(f"\nReceipt Classifier Results:")
    print(f"  Accuracy:  {receipt_metrics['accuracy']:.4f}")
    print(f"  Precision: {receipt_metrics['precision']:.4f}")
    print(f"  Recall:    {receipt_metrics['recall']:.4f}")
    print(f"  F1-Score:  {receipt_metrics['f1']:.4f}")
    
    # ========== PHASE 2: SIGNATURE DETECTION ==========
    print("\n[PHASE 2] Signature Detection")
    print("-" * 70)
    
    # Load signature dataset
    X_sig = load_images(SIGNATURES_PATH, (128, 128))
    X_nonsig = load_images(NON_SIGNATURES_PATH, (128, 128))
    
    if len(X_sig) == 0 or len(X_nonsig) == 0:
        print("⚠ No signature images found - creating synthetic dataset...")
        X_sig = np.random.rand(10, 128, 128, 3)
        X_nonsig = np.random.rand(10, 128, 128, 3)
    
    # Combine and label
    X_combined = np.vstack([X_sig, X_nonsig])
    y_combined = np.array([1] * len(X_sig) + [0] * len(X_nonsig))
    
    # Split
    X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(
        X_combined, y_combined, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y_combined
    )
    
    # Train CNN
    print("\nTraining CNN Model...")
    cnn_model = build_cnn_model()
    cnn_model.fit(X_train_sig, y_train_sig, validation_split=0.2, epochs=CNN_EPOCHS, verbose=0)
    
    cnn_pred = (cnn_model.predict(X_test_sig, verbose=0) > 0.5).astype(int).flatten()
    cnn_metrics = {
        'accuracy': accuracy_score(y_test_sig, cnn_pred),
        'precision': precision_score(y_test_sig, cnn_pred, zero_division=0),
        'recall': recall_score(y_test_sig, cnn_pred, zero_division=0),
        'f1': f1_score(y_test_sig, cnn_pred, zero_division=0),
    }
    
    print(f"CNN Results:")
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cnn_metrics['f1']:.4f}")
    
    # Train Transfer Learning
    print("\nTraining Transfer Learning Model (ResNet50)...")
    X_train_tl = np.array([smart_resize(img, (224, 224)) for img in X_train_sig])
    X_test_tl = np.array([smart_resize(img, (224, 224)) for img in X_test_sig])
    
    tl_model = build_transfer_learning_model()
    tl_model.fit(X_train_tl, y_train_sig, validation_split=0.2, epochs=TRANSFER_LEARNING_EPOCHS, verbose=0)
    
    tl_pred = (tl_model.predict(X_test_tl, verbose=0) > 0.5).astype(int).flatten()
    tl_metrics = {
        'accuracy': accuracy_score(y_test_sig, tl_pred),
        'precision': precision_score(y_test_sig, tl_pred, zero_division=0),
        'recall': recall_score(y_test_sig, tl_pred, zero_division=0),
        'f1': f1_score(y_test_sig, tl_pred, zero_division=0),
    }
    
    print(f"Transfer Learning Results:")
    print(f"  Accuracy:  {tl_metrics['accuracy']:.4f}")
    print(f"  Precision: {tl_metrics['precision']:.4f}")
    print(f"  Recall:    {tl_metrics['recall']:.4f}")
    print(f"  F1-Score:  {tl_metrics['f1']:.4f}")
    
    # ========== PHASE 3: RESULTS & VISUALIZATION ==========
    print("\n[PHASE 3] Results & Visualization")
    print("-" * 70)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Receipt Classifier Confusion Matrix
    cm_receipt = receipt_metrics['confusion_matrix']
    sns.heatmap(cm_receipt, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Incomplete', 'Complete'], yticklabels=['Incomplete', 'Complete'])
    axes[0, 0].set_title('Receipt Classifier\nConfusion Matrix')
    
    # All Metrics Comparison
    models = ['Receipt\nClassifier', 'CNN', 'Transfer\nLearning']
    accuracies = [receipt_metrics['accuracy'], cnn_metrics['accuracy'], tl_metrics['accuracy']]
    f1_scores = [receipt_metrics['f1'], cnn_metrics['f1'], tl_metrics['f1']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    axes[0, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Model Performance Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # CNN Confusion Matrix
    cm_cnn = confusion_matrix(y_test_sig, cnn_pred)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
                xticklabels=['Non-Sig', 'Signature'], yticklabels=['Non-Sig', 'Signature'])
    axes[1, 0].set_title('CNN Confusion Matrix')
    
    # Transfer Learning Confusion Matrix
    cm_tl = confusion_matrix(y_test_sig, tl_pred)
    sns.heatmap(cm_tl, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=['Non-Sig', 'Signature'], yticklabels=['Non-Sig', 'Signature'])
    axes[1, 1].set_title('Transfer Learning Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'results.png'), dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to results/results.png")
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': ['Receipt Classifier', 'Receipt Classifier', 'Receipt Classifier', 'Receipt Classifier',
                  'CNN', 'CNN', 'CNN', 'CNN',
                  'Transfer Learning', 'Transfer Learning', 'Transfer Learning', 'Transfer Learning'],
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 3,
        'Value': [
            receipt_metrics['accuracy'], receipt_metrics['precision'], receipt_metrics['recall'], receipt_metrics['f1'],
            cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1'],
            tl_metrics['accuracy'], tl_metrics['precision'], tl_metrics['recall'], tl_metrics['f1']
        ]
    })
    
    results_df.to_csv(os.path.join(RESULTS_PATH, 'metrics.csv'), index=False)
    print("✓ Metrics saved to results/metrics.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n📋 Receipt Classifier: {receipt_metrics['accuracy']:.2%} accuracy")
    print(f"🖼️  CNN Signature: {cnn_metrics['accuracy']:.2%} accuracy")
    print(f"🔄 Transfer Learning: {tl_metrics['accuracy']:.2%} accuracy")
    print(f"\n✓ Results saved to: {RESULTS_PATH}")
    print("="*70)


if __name__ == "__main__":
    run_pipeline()
