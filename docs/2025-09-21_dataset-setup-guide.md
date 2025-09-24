# 📊 Dataset Setup Guide for Training Script

## 🎯 Overview

To train the Indonesian ethnicity detection model, you need to organize your dataset in a specific directory structure.

## 📁 Required Directory Structure

Create the following folder structure in your project root:

```
etnis-id/
├── ml_training/
│   ├── script_training_fixed.py
│   └── DATASET_SETUP_GUIDE.md
├── dataset_periorbital/          ← CREATE THIS FOLDER
│   ├── Bugis/                   ← Bugis ethnicity images
│   ├── Sunda/                   ← Sundanese ethnicity images  
│   ├── Malay/                   ← Malay ethnicity images
│   ├── Jawa/                    ← Javanese ethnicity images
│   └── Banjar/                  ← Banjarese ethnicity images
└── model_ml/
    └── pickle_model.pkl         ← Will be created after training
```

## 📸 Image Requirements

### **Supported Formats:**
- `.jpg` / `.jpeg`
- `.png`

### **Recommended Specifications:**
- **Size**: Any size (script will resize to 400x200)
- **Quality**: Clear, well-lit face images
- **Content**: Periorbital region (eye area) or full face images
- **Quantity**: At least 50-100 images per ethnicity for good training

### **Image Organization:**
```
dataset_periorbital/
├── Bugis/
│   ├── bugis_001.jpg
│   ├── bugis_002.png
│   └── ...
├── Sunda/
│   ├── sunda_001.jpg
│   ├── sunda_002.png
│   └── ...
├── Malay/
│   ├── malay_001.jpg
│   ├── malay_002.png
│   └── ...
├── Jawa/
│   ├── jawa_001.jpg
│   ├── jawa_002.png
│   └── ...
└── Banjar/
    ├── banjar_001.jpg
    ├── banjar_002.png
    └── ...
```

## 🚀 Running the Training Script

### **1. Setup Dataset:**
```bash
# Create the dataset directory
mkdir dataset_periorbital
mkdir dataset_periorbital/Bugis
mkdir dataset_periorbital/Sunda  
mkdir dataset_periorbital/Malay
mkdir dataset_periorbital/Jawa
mkdir dataset_periorbital/Banjar

# Copy your images to respective folders
```

### **2. Run Training:**
```bash
# Navigate to the project directory
cd D:\Projects\game-issat\etnis-id

# Run the fixed training script
python ml_training/script_training_fixed.py
```

## 📊 Expected Output

When training runs successfully, you should see:

```
🚀 STARTING INDONESIAN ETHNICITY RECOGNITION TRAINING
======================================================================
📊 PHASE 1: LOADING DATA
------------------------------
📁 Loading data from: dataset_periorbital
📂 Found classes: ['Banjar', 'Bugis', 'Jawa', 'Malay', 'Sunda']
📸 Bugis: 150 images
📸 Sunda: 200 images
📸 Malay: 180 images
📸 Jawa: 220 images
📸 Banjar: 160 images
✅ Total loaded: 910 images

🔄 PHASE 2: PREPROCESSING
------------------------------
🔄 Preprocessing GLCM: Converting RGB to Grayscale...
✅ Converted 910 images to grayscale
🔄 Preprocessing Color: Converting RGB to HSV...
✅ Converted 910 images to HSV

🧮 PHASE 3: FEATURE EXTRACTION
------------------------------
🧮 Extracting GLCM features...
✅ GLCM features extracted: 910 samples, 20 features each
🎨 Extracting Color Histogram features...
✅ Color features extracted: 910 samples, 32 features each

🔗 PHASE 4: COMBINING FEATURES
------------------------------
Combined features shape: (910, 52)
GLCM features: 20
Color features: 32
Total features: 52

🎯 PHASE 5: MODEL TRAINING
------------------------------
🔄 Running 6-fold cross validation...
✅ Cross-validation results:
   Mean accuracy: 85.23% ± 3.45%
   Individual fold scores: [87.5 83.2 85.1 84.8 86.3 84.9]
🤖 Training final Random Forest model...

💾 PHASE 6: SAVING MODEL
------------------------------
💾 Saving model to model_ml/pickle_model.pkl...
✅ Model saved successfully!

🎉 TRAINING COMPLETED SUCCESSFULLY!
======================================================================
📊 Final Model Performance:
   - Mean CV Accuracy: 85.23%
   - Standard Deviation: 3.45%
   - Features used: 52 (GLCM: 20, Color: 32)
   - Model saved to: model_ml/pickle_model.pkl
   - Supported ethnicities: ['Bugis', 'Sunda', 'Malay', 'Jawa', 'Banjar']
```

## ⚠️ Troubleshooting

### **Error: "Dataset directory not found"**
- Make sure `dataset_periorbital` folder exists in project root
- Check folder spelling and location

### **Error: "No valid images found"**
- Ensure images are in correct format (.jpg, .jpeg, .png)
- Check that images are not corrupted
- Verify images are placed in correct ethnicity folders

### **Error: "Memory error"**
- Reduce number of images per class
- Use smaller image sizes
- Close other applications to free up RAM

### **Low Accuracy (< 70%)**
- Add more training images per ethnicity
- Ensure images are high quality and representative
- Check that images are correctly labeled

## 📈 Performance Tips

1. **Balanced Dataset**: Try to have similar number of images per ethnicity
2. **Quality Images**: Use clear, well-lit face images
3. **Representative Samples**: Include diverse samples for each ethnicity
4. **Sufficient Data**: At least 100+ images per class for good performance

## 🔄 After Training

Once training completes successfully:

1. **Model File**: `model_ml/pickle_model.pkl` will be created
2. **Test the Model**: Run your existing test scripts to verify the model works
3. **Update Server**: The ML server will automatically use the new model
4. **Deploy**: Your system is ready for production use!

---

**Happy Training! 🎉**
