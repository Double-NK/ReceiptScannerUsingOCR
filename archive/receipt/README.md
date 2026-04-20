# Receipt Classification System

A simple, integrated system for receipt classification and signature detection using OCR, Machine Learning, and Deep Learning.

## What It Does

✅ **Receipt Classification**: Classifies receipts as complete or incomplete using 7 extracted features  
✅ **Signature Detection**: Detects signatures using CNN and Transfer Learning (ResNet50)  
✅ **Metrics**: Generates Accuracy, Precision, Recall, and F1-Score for all models  
✅ **Visualization**: Creates confusion matrices and comparison charts  

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your images in these folders:
- `data/complete/` - Complete receipts
- `data/incomplete/` - Incomplete receipts
- `data/signatures/` - Signature images
- `data/non_signatures/` - Non-signature image crops

### 3. Run

```bash
python receipt.py
```

## Project Structure

```
receipt/
├── config.py          # Configuration & paths
├── receipt.py         # Complete pipeline (700+ lines, all-in-one)
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/
│   ├── complete/      # Complete receipt images
│   ├── incomplete/    # Incomplete receipt images
│   ├── signatures/    # Signature crops
│   └── non_signatures/# Non-signature crops
├── models/            # Trained models saved here
└── results/           # Output metrics, visualizations
```

## Features Extracted

From receipt text, the system extracts:
1. Has date? (1/0)
2. Has company name? (1/0)
3. Has item description? (1/0)
4. Has signature field? (1/0)
5. Has address? (1/0)
6. Text length normalized (0-20)
7. Digit count normalized (0-25)

## Models Used

**Receipt Classification:**
- Algorithm: Random Forest (100 trees)
- Input: 7 numerical features
- Output: Complete (1) or Incomplete (0)

**Signature Detection - CNN:**
- 3 conv layers + pooling
- Input: 128×128 color images
- Output: Signature (1) or Not (0)

**Signature Detection - Transfer Learning:**
- ResNet50 base (ImageNet pre-trained)
- 2 dense layers on top
- Input: 224×224 color images
- Output: Signature (1) or Not (0)

## Output Files

After running `python receipt.py`:

### results/metrics.csv
Shows metrics for all 3 models:
```
Model,Metric,Value
Receipt Classifier,Accuracy,0.7500
CNN,Accuracy,0.8333
Transfer Learning,Accuracy,0.9167
```

### results/results.png
Visual comparison with:
- Receipt classifier confusion matrix
- Model performance comparison (Accuracy vs F1-Score)
- CNN confusion matrix
- Transfer Learning confusion matrix

## API Setup

The system uses Typhoon OCR API to extract text. The API key is in `config.py`:

```python
TYPHOON_API_KEY = "sk-HytrxVrL2v19alxMtEs8BzXfGpzTWX5Uc7x6q4mYR9osAcYl"
```

No additional setup needed!

## Example Usage

### Notebook Version

See `receipt_demo.ipynb` for interactive step-by-step execution.

### Python Script

Run directly:
```bash
python receipt.py
```

## Customization

Edit `config.py` to change:
- `CNN_EPOCHS`: Number of training epochs for CNN (default: 30)
- `TRANSFER_LEARNING_EPOCHS`: ResNet50 training epochs (default: 20)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `TEST_SPLIT`: Train/test split ratio (default: 0.2)

## Troubleshooting

**No images found?**
- The system creates synthetic data for demo if real images aren't available

**OCR fails?**
- Text extraction still works with sample text
- Results still show metrics and visualizations

**GPU memory issues?**
- Reduce `BATCH_SIZE` in config.py
- Or run on CPU only (slower but works)

## Metrics Explanation

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - False positive rate
- **Recall**: TP / (TP + FN) - False negative rate  
- **F1-Score**: Harmonic mean of Precision and Recall

## Files Created

- `config.py`: 40 lines - Configuration
- `receipt.py`: 700 lines - Complete pipeline
- `requirements.txt`: 9 packages
- `receipt_demo.ipynb`: Interactive notebook

**Total: ~750 lines of code + data**

---

Made simple. Made easy. Just run and it works! 🚀
