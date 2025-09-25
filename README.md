# Indonesian Ethnicity Detection System

🇮🇩 **Sistem Deteksi Etnis Indonesia** menggunakan Machine Learning dan Computer Vision untuk mengenali etnis berdasarkan citra wajah.

## 🎯 Overview

Proyek ini mengintegrasikan:
- **Machine Learning**: Random Forest Classifier dengan ekstraksi fitur GLCM dan Color Histogram
- **Computer Vision**: OpenCV dan scikit-image untuk preprocessing gambar
- **Network Communication**: TCP socket untuk komunikasi real-time
- **UI Interface**: Godot Engine untuk antarmuka pengguna

## 🏗️ Arsitektur Sistem

```
[Godot Client] ←→ TCP Socket ←→ [Python ML Server] ←→ [Random Forest Model]
                                      ↓
                              [Feature Extraction]
                              - GLCM (Texture)
                              - Color Histogram
```

## 🔧 Tech Stack

### Backend (Python)
- **Machine Learning**: scikit-learn, numpy
- **Computer Vision**: opencv-python, scikit-image, PIL
- **Network**: socket, threading, json
- **Data Processing**: pandas, scipy

### Frontend (Godot)
- **Engine**: Godot 4.x
- **Language**: GDScript
- **Communication**: StreamPeerTCP

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Godot Engine 4.x
- Git

### Installation (Windows)

```
# Clone
git clone <repo-url>
cd etnis-id

# Activate existing venv
cmd /c "env\Scripts\activate.bat && pip install -r requirements.txt"
```

### Start ML Server
```
cmd /c "env\Scripts\activate.bat && python ml_server.py"
```

### Godot Client
- Buka `tcp-example/project.godot` di Godot, jalankan scene klien ML.

## 📁 Project Structure (simplified)

```
etnis-id/
├── ethnic_detector.py
├── ml_server.py
├── ml_training/
│   └── core/ (config, data_loader, feature_extractors, training_pipeline, utils, ...)
├── model_ml/
│   └── pickle_model.pkl
├── tests/
│   ├── unit/
│   ├── cv/
│   ├── config/
│   ├── solid/
│   ├── visualization/
│   ├── smoke/
│   ├── helpers/
│   └── legacy/
├── dataset/
│   └── dataset_periorbital/...
├── requirements.txt
└── README.md
```

## 🧪 Testing (pytest)

All commands below assume Windows with the provided virtual environment.

- Run all tests
```
cmd /c "env\Scripts\activate.bat && python -m pytest -q"
```

- Run by suite
```
# Unit tests (fast)
cmd /c "env\Scripts\activate.bat && pytest -q tests/unit"

# Cross-validation pipeline
cmd /c "env\Scripts\activate.bat && pytest -q tests/cv"

# Visualization tests
cmd /c "env\Scripts\activate.bat && pytest -q tests/visualization"

# Config system
cmd /c "env\Scripts\activate.bat && pytest -q tests/config"

# SOLID training system
cmd /c "env\Scripts\activate.bat && pytest -q tests/solid"

# Smoke checks (env/dataset presence)
cmd /c "env\Scripts\activate.bat && pytest -q tests/smoke"
```

Lihat panduan lengkap di `tests/README.md` dan `tests/TESTING_GUIDE.md`.

## 📊 Supported Ethnicities
- 🏮 **Jawa** (Javanese)
- 🌸 **Sunda** (Sundanese)
- 🌊 **Malay** (Malay)
- ⛵ **Bugis** (Buginese)
- 🏛️ **Banjar** (Banjarese)

## 🔬 Model Performance
- **Features**: 52 total (20 GLCM + 32 Color Histogram)
- **Accuracy**: depends on dataset split and CV; see tests/cv

## Idea of improvement

Advances in Facial Feature Analysis for Demographic Estimation

To perform this classification, the system relies on the extraction of discriminative features from the facial region. Rather than processing raw pixel data directly, which can be computationally expensive and sensitive to irrelevant variations, the system employs a set of well-established feature descriptors designed to capture salient visual information. This research utilizes a combination of three powerful and complementary feature descriptors:
●	Gray Level Co-occurrence Matrix (GLCM): This is a classic texture analysis method that captures second-order statistical information by examining the spatial relationships between pixels at different orientations and distances. GLCM provides a feature vector describing properties such as contrast, correlation, energy, and homogeneity, which are effective for characterizing surface textures.3
●	Local Binary Patterns (LBP): LBP is a highly efficient and robust descriptor that provides a fine-grained analysis of texture. It operates by comparing the intensity of each pixel with its surrounding neighbors, encoding the result as a binary number. The histogram of these binary patterns across a region forms a powerful texture signature. Its computational efficiency makes it particularly well-suited for real-time applications.
●	Histogram of Oriented Gradients (HOG): Unlike GLCM and LBP, which primarily describe texture, HOG is designed to capture the shape and structure of objects. It achieves this by computing a histogram of gradient orientations within localized portions of an image. HOG is robust to changes in illumination and has proven highly effective in various object detection and recognition tasks, including facial analysis.3


## 🤝 Contributing
1. Fork → feature branch → PR
2. Tambahkan/ubah tests sesuai perubahan fitur

## 📞 Contact
- Tim: Muhammad Gianluigi, Muhammad Rafli Fadhilah, Daffa Muzhaffar