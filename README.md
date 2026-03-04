# 🔬 Evaluasi Robustness Model Sentimen Bahasa Indonesia terhadap Noise Sintetis Menggunakan Arsitektur Hybrid Word-Character Lightweight

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.10-red?logo=keras)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📌 Deskripsi Proyek

Penelitian ini membangun dan mengevaluasi sistem klasifikasi sentimen Bahasa Indonesia yang **tahan terhadap noise tipografi** menggunakan arsitektur **Hybrid Word-Character Lightweight**. Model diuji pada 4 level noise sintetis khas Twitter Indonesia (0%, 10%, 20%, 30%) dan dibandingkan dengan model baseline BiLSTM word-only.

**Hasil utama:** Hybrid model mengurangi drop rate accuracy sebesar **58.82%** pada noise level 30% (dari 3.06% menjadi 1.26%) dibandingkan baseline, dengan Robustness Score **0.9874** vs **0.9694**.

---

## 🎯 Tujuan Penelitian

1. Mengukur dampak kuantitatif noise tipografi terhadap performa model sentimen Bahasa Indonesia
2. Mengembangkan arsitektur hybrid lightweight yang menggabungkan word-level dan character-level features
3. Menganalisis degradasi performa secara kuantitatif pada berbagai tingkat noise
4. Memvalidasi peningkatan robustness melalui uji statistik (Paired t-test & Wilcoxon Signed-Rank)
5. Menyusun framework evaluasi robustness untuk NLP Bahasa Indonesia

---

## 📊 Dataset

| Properti | Detail |
|---|---|
| Nama | Twitter Dataset PPKM Indonesia |
| Sumber | [Kaggle — anggapurnama/twitter-dataset-ppkm](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm) |
| Total data | 23.644 tweets |
| Kelas | 3 (Positif: 74.9%, Netral: 16.8%, Negatif: 8.3%) |
| Bahasa | Bahasa Indonesia (informal/Twitter) |
| Split | Train 70% / Val 15% / Test 15% (Stratified) |

---

## 🔧 Noise Injection Engine

6 jenis noise khas Bahasa Indonesia disimulasikan secara programatik:

| # | Jenis Noise | Contoh | Bobot |
|---|---|---|---|
| 1 | Char Substitution | `bagus → baguz` | 20% |
| 2 | Vowel Removal | `mantap → mntp` | 20% |
| 3 | Slang Replacement | `tidak → gak` | 25% |
| 4 | Char Repetition | `parah → parahhh` | 15% |
| 5 | Keyboard Proximity | `makan → majan` | 10% |
| 6 | Char Deletion | `sekarang → sekrang` | 10% |

Noise diterapkan **hanya pada test set** dengan 4 versi: clean, noise 10%, noise 20%, noise 30%.

---

## 🏗️ Arsitektur Model

### Baseline Model — Word BiLSTM

```
Word Input (max_len=100)
    ↓
Word Embedding (20,000 × 128, trainable)
    ↓
SpatialDropout1D
    ↓
Bidirectional LSTM (64 units, cuDNN optimized)
    ↓
GlobalMaxPooling1D
    ↓
Dense (128) → Dropout → Dense (64) → Dropout
    ↓
Dense (3) + Softmax

Total params: 2,683,779
```

### Hybrid Model — Word-Character Fusion

```
Word Input (max_len=100)        Char Input (max_len=300)
        ↓                               ↓
Word Embedding (128-dim)        Char Embedding (64-dim)
        ↓                               ↓
SpatialDropout1D                SpatialDropout1D
        ↓                               ↓
Bidirectional LSTM (64 units)   Conv1D (64 filters, kernel=3)
        ↓                               ↓
GlobalMaxPooling1D              BatchNormalization
                                        ↓
                                Conv1D (128 filters, kernel=3)
                                        ↓
                                GlobalMaxPooling1D
                                        ↓
                    ┌───────────────────┘
                    ↓
            Concatenate [word_feat + char_feat]
                    ↓
            LayerNormalization
                    ↓
        Dense (256) → Dropout → Dense (128) → Dropout
                    ↓
              Dense (64) → Dropout
                    ↓
            Dense (3) + Softmax

Total params: 2,806,787
```

---

## 📈 Hasil Evaluasi

### Robustness Drop Rate

| Noise Level | Baseline Acc | Hybrid Acc | Δ Acc | Base Drop% | Hybrid Drop% |
|---|---|---|---|---|---|
| Clean (0%) | 0.8243 | 0.8195 | −0.0048 | 0.00% | 0.00% |
| Noise 10% | 0.8203 | 0.8157 | −0.0046 | 0.49% | 0.46% |
| Noise 20% | 0.8129 | 0.8140 | **+0.0011** | 1.38% | 0.67% |
| Noise 30% | 0.7991 | 0.8092 | **+0.0101** | 3.06% | **1.26%** |

### Macro F1 Score

| Noise Level | Baseline F1 | Hybrid F1 | Δ F1 |
|---|---|---|---|
| Clean (0%) | 0.7197 | 0.6861 | −0.0336 |
| Noise 10% | 0.7101 | 0.6773 | −0.0328 |
| Noise 20% | 0.6971 | 0.6651 | −0.0320 |
| Noise 30% | 0.6817 | 0.6544 | −0.0273 |

### Robustness Score

| Model | Robustness Score | Drop Rate @30% |
|---|---|---|
| Baseline BiLSTM | 0.9694 | 3.06% |
| Hybrid Word-Char | **0.9874** | **1.26%** |
| **Peningkatan** | **+1.85%** | **−58.82%** |

---

## 📁 Struktur Project

```
sentiment-robustness-id/
│
├── 📂 data/
│   ├── raw/
│   │   └── INA_TweetsPPKM_Labeled_Pure.csv   # dataset asli
│   ├── processed/
│   │   └── data_clean.csv                     # hasil preprocessing + split
│   └── noisy/
│       ├── test_clean.csv                     # test set bersih
│       ├── test_noise_10.csv                  # test set noise 10%
│       ├── test_noise_20.csv                  # test set noise 20%
│       └── test_noise_30.csv                  # test set noise 30%
│
├── 📂 notebooks/
│   ├── 00_Setup_Project.ipynb                 # setup folder & upload dataset
│   ├── 01_EDA.ipynb                           # exploratory data analysis
│   ├── 02_Preprocessing.ipynb                 # cleaning & stratified split
│   ├── 03_Noise_Injection.ipynb               # noise engine & generate test sets
│   ├── 04_Baseline_Model.ipynb                # train & eval baseline BiLSTM
│   ├── 05_Hybrid_Model.ipynb                  # train & eval hybrid model
│   └── 06_Evaluation.ipynb                    # analisis statistik & final report
│
├── 📂 src/
│   ├── config.py                              # central configuration (path & hyperparams)
│   └── noise_injection.py                     # noise engine module (reusable)
│
├── 📂 models/
│   ├── baseline_model.h5                      # trained baseline model
│   ├── hybrid_model.keras                     # trained hybrid model
│   ├── word_tokenizer.pkl                     # word-level tokenizer
│   └── char_tokenizer.pkl                     # character-level tokenizer
│
├── 📂 results/
│   ├── metrics_baseline.csv                   # metrik evaluasi baseline
│   ├── metrics_hybrid.csv                     # metrik evaluasi hybrid
│   ├── final_metrics_table.csv                # tabel perbandingan lengkap
│   ├── statistical_tests.csv                  # hasil paired t-test & wilcoxon
│   ├── effect_size.csv                        # cohen's d effect size
│   ├── ablation_study.csv                     # ablation study results
│   ├── baseline_training_log.csv              # log training baseline per epoch
│   ├── hybrid_training_log.csv                # log training hybrid per epoch
│   ├── final_report.xlsx                      # excel report multi-sheet
│   └── plots/
│       ├── 01_label_distribution.png
│       ├── 01_char_length.png
│       ├── 01_word_length.png
│       ├── 01_noise_natural.png
│       ├── 01_wordcloud.png
│       ├── 01_top_words.png
│       ├── 04_baseline_architecture.png
│       ├── 04_baseline_learning_curve.png
│       ├── 04_baseline_confusion_matrix.png
│       ├── 04_baseline_robustness_curve.png
│       ├── 05_hybrid_architecture.png
│       ├── 05_learning_curve_comparison.png
│       ├── 05_hybrid_confusion_matrix.png
│       ├── 05_robustness_curve_comparison.png
│       ├── 06_robustness_curve_final.png      # ⭐ plot utama paper
│       ├── 06_drop_rate_comparison.png        # ⭐ plot utama paper
│       └── 06_perclass_f1_heatmap.png         # ⭐ plot utama paper
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗂️ Output Penting per Notebook

### `01_EDA.ipynb`
| Output | Lokasi | Kegunaan |
|---|---|---|
| `01_label_distribution.png` | `results/plots/` | Distribusi kelas untuk Bab 3 |
| `01_word_length.png` | `results/plots/` | Justifikasi MAX_SEQ_LEN |
| `01_noise_natural.png` | `results/plots/` | Motivasi noise injection |
| `01_wordcloud.png` | `results/plots/` | Visualisasi karakteristik data |

### `02_Preprocessing.ipynb`
| Output | Lokasi | Kegunaan |
|---|---|---|
| `data_clean.csv` | `data/processed/` | Input untuk semua notebook berikutnya |

### `03_Noise_Injection.ipynb`
| Output | Lokasi | Kegunaan |
|---|---|---|
| `test_clean.csv` | `data/noisy/` | Test set baseline evaluasi |
| `test_noise_10/20/30.csv` | `data/noisy/` | Test set untuk robustness eval |
| `noise_injection.py` | `src/` | Modul reusable untuk penelitian lanjutan |

### `04_Baseline_Model.ipynb`
| Output | Lokasi | Kegunaan |
|---|---|---|
| `baseline_model.h5` | `models/` | Model baseline untuk komparasi |
| `word_tokenizer.pkl` | `models/` | Dipakai ulang di notebook 05 & 06 |
| `metrics_baseline.csv` | `results/` | Angka drop rate baseline |
| `baseline_training_log.csv` | `results/` | Dipakai di learning curve comparison |
| `04_baseline_learning_curve.png` | `results/plots/` | Analisis konvergensi |
| `04_baseline_confusion_matrix.png` | `results/plots/` | Analisis per kelas |
| `04_baseline_robustness_curve.png` | `results/plots/` | Visualisasi degradasi |

### `05_Hybrid_Model.ipynb`
| Output | Lokasi | Kegunaan |
|---|---|---|
| `hybrid_model.keras` | `models/` | Model utama penelitian |
| `char_tokenizer.pkl` | `models/` | Dipakai di notebook 06 |
| `metrics_hybrid.csv` | `results/` | Angka drop rate hybrid |
| `05_learning_curve_comparison.png` | `results/plots/` | Perbandingan konvergensi |
| `05_robustness_curve_comparison.png` | `results/plots/` | Preview perbandingan awal |

### `06_Evaluation.ipynb` ⭐
| Output | Lokasi | Kegunaan |
|---|---|---|
| `final_metrics_table.csv` | `results/` | Tabel utama paper |
| `statistical_tests.csv` | `results/` | Bukti signifikansi statistik |
| `effect_size.csv` | `results/` | Cohen's d untuk paper |
| `ablation_study.csv` | `results/` | Kontribusi tiap komponen |
| `final_report.xlsx` | `results/` | Laporan lengkap multi-sheet |
| `06_robustness_curve_final.png` | `results/plots/` | ⭐ Gambar utama paper |
| `06_drop_rate_comparison.png` | `results/plots/` | ⭐ Gambar utama paper |
| `06_perclass_f1_heatmap.png` | `results/plots/` | ⭐ Gambar utama paper |

---

## 🚀 Cara Menjalankan

### Prasyarat

```bash
# Clone repository
git clone https://github.com/USERNAME/sentiment-robustness-id.git
cd sentiment-robustness-id

# Install dependencies
pip install -r requirements.txt
```

### Urutan Eksekusi Notebook

> ⚠️ Jalankan notebook **secara berurutan**. Setiap notebook bergantung pada output notebook sebelumnya.

| Urutan | Notebook | Runtime | Estimasi Waktu |
|---|---|---|---|
| 1 | `00_Setup_Project.ipynb` | CPU | ~2 menit |
| 2 | `01_EDA.ipynb` | CPU | ~5 menit |
| 3 | `02_Preprocessing.ipynb` | CPU | ~3 menit |
| 4 | `03_Noise_Injection.ipynb` | CPU | ~5 menit |
| 5 | `04_Baseline_Model.ipynb` | **GPU T4** | ~10 menit |
| 6 | `05_Hybrid_Model.ipynb` | **GPU T4** | ~15 menit |
| 7 | `06_Evaluation.ipynb` | CPU | ~10 menit |

### Aktifkan GPU di Google Colab

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

> Aktifkan GPU **hanya sebelum** menjalankan notebook 04 dan 05.

---

## ⚙️ Konfigurasi

Semua hyperparameter dan path terpusat di `src/config.py`:

```python
# Model Hyperparameters
MAX_WORDS       = 20000     # vocab size word-level
MAX_CHARS       = 100       # vocab size char-level
MAX_SEQ_LEN     = 100       # panjang sequence word
MAX_CHAR_LEN    = 300       # panjang sequence char
EMBEDDING_DIM   = 128       # dimensi word embedding
CHAR_EMBED_DIM  = 64        # dimensi char embedding
LSTM_UNITS      = 64        # BiLSTM units
CNN_FILTERS     = 64        # CNN filters (char branch)
CNN_KERNEL_SIZE = 3         # CNN kernel size
DROPOUT_RATE    = 0.3       # dropout rate
BATCH_SIZE      = 64        # batch size training
EPOCHS          = 20        # max epochs (early stopping)
LEARNING_RATE   = 0.001     # Adam learning rate

# Noise Configuration
NOISE_LEVELS    = [0.10, 0.20, 0.30]

# Split Configuration
TEST_SIZE       = 0.15
VAL_SIZE        = 0.15
RANDOM_STATE    = 42
```

---

## 📦 Dependencies

```
tensorflow==2.19.0
keras==3.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
wordcloud>=1.9.0
openpyxl>=3.1.0
```

Install semua sekaligus:
```bash
pip install -r requirements.txt
```

---

## 📂 Yang Di-push ke GitHub

File dan folder berikut **disertakan** dalam repository:

```
✅ notebooks/          — semua notebook (.ipynb)
✅ src/                — config.py & noise_injection.py
✅ results/plots/      — semua visualisasi (PNG)
✅ results/*.csv       — semua tabel metrik
✅ results/*.xlsx      — final report Excel
✅ requirements.txt
✅ README.md
✅ .gitignore
```

File berikut **tidak di-push** (tercantum di `.gitignore`):

```
❌ data/               — dataset terlalu besar untuk GitHub
❌ models/             — file model besar (.h5, .keras, .pkl)
```

> 💡 Dataset tersedia di [Kaggle](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm). Model dapat di-generate ulang dengan menjalankan notebook secara berurutan.

---

## 📐 Kontribusi Ilmiah

1. **Framework evaluasi robustness** untuk model NLP Bahasa Indonesia terhadap noise tipografi
2. **Noise modeling** khas Bahasa Indonesia (slang, singkatan, typo Twitter)
3. **Arsitektur hybrid lightweight** Word-Character yang terbukti lebih robust tanpa GPU besar
4. **Analisis degradasi kuantitatif** dengan validasi statistik (Paired t-test + Wilcoxon + Cohen's d)
5. **Benchmark dataset** evaluasi robustness sentimen Twitter Indonesia

---

## 📚 Referensi

- Twitter Dataset PPKM: [Kaggle — anggapurnama](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm)
- IndoNLU Benchmark: [IndoNLU](https://github.com/IndoNLP/indonlu)
- Character-level CNN for NLP: Zhang et al. (2015)
- Robustness evaluation NLP: Belinkov & Bisk (2018)

---

## 👤 Author

**[Fatahillah]**

---

<div align="center">
  <sub>Built with ❤️ for Indonesian NLP Research</sub>
</div>