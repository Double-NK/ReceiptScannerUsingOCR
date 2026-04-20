"""
Configuration for Receipt Classification System
"""

import os

# API Configuration
TYPHOON_API_KEY = "sk-HytrxVrL2v19alxMtEs8BzXfGpzTWX5Uc7x6q4mYR9osAcYl"

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
COMPLETE_PATH = os.path.join(DATA_PATH, "complete")
INCOMPLETE_PATH = os.path.join(DATA_PATH, "incomplete")
SIGNATURES_PATH = os.path.join(DATA_PATH, "signatures")
NON_SIGNATURES_PATH = os.path.join(DATA_PATH, "non_signatures")
MODELS_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

# Create directories
for path in [COMPLETE_PATH, INCOMPLETE_PATH, SIGNATURES_PATH, NON_SIGNATURES_PATH, MODELS_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

# Training Parameters
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Model Parameters
CNN_EPOCHS = 30
TRANSFER_LEARNING_EPOCHS = 20
BATCH_SIZE = 16
