# =============================================================
# config.py — Central Configuration
# Evaluasi Robustness Model Sentimen Bahasa Indonesia
# =============================================================

import os

# ROOT
PROJECT_NAME = "sentiment-robustness-id"
ROOT = "/content/drive/MyDrive/sentiment-robustness-id"

# DATA PATHS
RAW_DATA_PATH      = os.path.join(ROOT, "data/raw/INA_TweetsPPKM_Labeled_Pure.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT, "data/processed/data_clean.csv")
NOISY_DIR          = os.path.join(ROOT, "data/noisy")

# NOISY TEST SET PATHS
TEST_CLEAN_PATH    = os.path.join(ROOT, "data/noisy/test_clean.csv")
TEST_NOISE_10_PATH = os.path.join(ROOT, "data/noisy/test_noise_10.csv")
TEST_NOISE_20_PATH = os.path.join(ROOT, "data/noisy/test_noise_20.csv")
TEST_NOISE_30_PATH = os.path.join(ROOT, "data/noisy/test_noise_30.csv")

# MODEL PATHS
BASELINE_MODEL_PATH = os.path.join(ROOT, "models/baseline_model.h5")
HYBRID_MODEL_PATH   = os.path.join(ROOT, "models/hybrid_model.h5")

# RESULTS
RESULTS_DIR        = os.path.join(ROOT, "results")
PLOTS_DIR          = os.path.join(ROOT, "results/plots")
METRICS_TABLE_PATH = os.path.join(ROOT, "results/metrics_table.csv")

# DATASET CONFIG
TEXT_COLUMN   = "Tweet"
LABEL_COLUMN  = "sentiment"
NUM_CLASSES   = 3           # 0=Negatif, 1=Positif, 2=Netral
LABEL_MAP     = {0: "Negatif", 1: "Positif", 2: "Netral"}

# SPLIT CONFIG
TEST_SIZE       = 0.15
VAL_SIZE        = 0.15
RANDOM_STATE    = 42

# NOISE CONFIG
NOISE_LEVELS    = [0.10, 0.20, 0.30]

# MODEL HYPERPARAMETERS
MAX_WORDS       = 20000     # vocab size word-level
MAX_CHARS       = 100       # vocab size char-level
MAX_SEQ_LEN     = 100       # panjang sequence word
MAX_CHAR_LEN    = 300       # panjang sequence char
EMBEDDING_DIM   = 128
CHAR_EMBED_DIM  = 64
LSTM_UNITS      = 64
CNN_FILTERS     = 64
CNN_KERNEL_SIZE = 3
DROPOUT_RATE    = 0.3
BATCH_SIZE      = 64
EPOCHS          = 20
LEARNING_RATE   = 0.001
