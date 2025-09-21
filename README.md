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
- Python 3.8+
- Godot Engine 4.x
- Git

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/indonesian-ethnicity-detection.git
cd indonesian-ethnicity-detection
```

2. **Setup Python Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Start ML Server**
```bash
python ml_server.py
```

4. **Open Godot Project**
```bash
# Open tcp-example/project.godot with Godot Engine
```

### Usage

1. **Start ML Server**: Run `python ml_server.py`
2. **Open Godot Client**: Launch the Godot project
3. **Connect**: Client akan otomatis connect ke server
4. **Upload Image**: Pilih gambar untuk deteksi etnis
5. **View Results**: Lihat hasil prediksi dan confidence score

## 📊 Supported Ethnicities

- 🏮 **Jawa** (Javanese)
- 🌸 **Sunda** (Sundanese) 
- 🌊 **Malay** (Malay)
- ⛵ **Bugis** (Buginese)
- 🏛️ **Banjar** (Banjarese)

## 🔬 Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 52 total (20 GLCM + 32 Color Histogram)
- **Training Data**: Indonesian ethnic faces dataset
- **Accuracy**: ~85% (varies by ethnicity)

## 📁 Project Structure

```
proyek_etnis/
├── 🤖 ethnic_detector.py      # Core ML engine
├── 🌐 ml_server.py           # TCP server
├── 📊 script_training.py     # Model training script
├── 🎮 tcp-example/           # Godot client project
├── 🧠 model_ml/             # Trained ML models
│   └── pickle_model.pkl
├── 🧪 tests/                # Testing scripts
├── 📋 requirements.txt      # Python dependencies
└── 📖 README.md            # This file
```

## 🧪 Testing

Run comprehensive tests:
```bash
# Test all components
python integration_test.py

# Test individual components
python test_dependencies.py
python test_ml_model.py
python tcp_test_client.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🙏 Acknowledgments

- Dataset courtesy of Indonesian ethnic faces research Telkom University
- Built with scikit-learn and Godot Engine
- Inspired by computer vision research in ethnic recognition

## 📞 Contact

- **Author**: - Muhammad Gianluigi 
              - Muhammad Rafli Fadhilah
              - Daffa Muzhaffar
---