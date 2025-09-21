# 🧪 Panduan Testing Sistem Ethnic Detection

## 📋 Daftar Isi
1. [Overview Testing](#overview)
2. [Persiapan Testing](#persiapan)
3. [Test Individual](#test-individual)
4. [Test Integration](#test-integration)
5. [Troubleshooting](#troubleshooting)
6. [Checklist Verification](#checklist)

---

## 🎯 Overview Testing {#overview}

Sistem testing ini dirancang untuk memverifikasi bahwa:
- ✅ **Dependencies** terinstall dengan benar
- ✅ **ML Model** berfungsi dan menggunakan model asli (bukan simulasi)
- ✅ **TCP Communication** antara server dan client bekerja
- ✅ **End-to-End Workflow** dari Godot ke ML Server berjalan sempurna

### 📁 File Testing yang Dibuat:
```
test_dependencies.py    # Test dependencies & environment
test_ml_model.py       # Test ML model langsung
tcp_test_client.py     # Test TCP communication tanpa Godot
integration_test.py    # Test comprehensive end-to-end
```

---

## 🔧 Persiapan Testing {#persiapan}

### 1. **Install Dependencies**
```bash
# Pastikan di direktori proyek
cd d:\ISSAT_PCD\ProyekEtnis\proyek_etnis

# Install semua dependencies
pip install -r requirements.txt

# Atau install manual:
pip install scikit-learn>=1.0.0 numpy>=1.21.0 opencv-python>=4.5.0 scikit-image>=0.18.0 Pillow>=8.0.0 scipy>=1.7.0
```

### 2. **Verifikasi File Model**
```bash
# Check model file exists
dir model_ml\pickle_model.pkl
```

### 3. **Persiapan Terminal**
```bash
# Buka 2 terminal/command prompt:
# Terminal 1: Untuk menjalankan ML Server
# Terminal 2: Untuk menjalankan test scripts
```

---

## 🧪 Test Individual {#test-individual}

### Test 1: **Dependencies & Environment**
```bash
python test_dependencies.py
```

**Expected Output:**
```
✅ scikit-learn: 1.3.0
✅ numpy: 1.24.0
✅ opencv-python: 4.8.0
✅ scikit-image: 0.20.0
✅ Pillow: 9.5.0
✅ scipy: 1.10.0
✅ Model file found: model_ml/pickle_model.pkl
✅ Model loaded successfully
🎯 OVERALL STATUS: ✅ READY FOR ML PROCESSING
```

**Jika GAGAL:**
- Install missing dependencies: `pip install <package_name>`
- Check Python version compatibility
- Verify model file exists

---

### Test 2: **ML Model Direct**
```bash
python test_ml_model.py
```

**Expected Output:**
```
🤖 Initializing ML Ethnic Detector...
✅ ML Model berhasil dimuat!

🧮 TESTING FEATURE EXTRACTION
✅ Feature extraction successful: Shape: (1, 52)

🎯 TESTING ETHNIC PREDICTIONS
✅ Prediction successful: Ethnicity: Jawa, Confidence: 87.3%

📦 TESTING BASE64 WORKFLOW
✅ Base64 prediction successful: Ethnicity: Sunda, Confidence: 82.1%

🔄 TESTING PREDICTION CONSISTENCY
📊 Consistency rate: 100.0%

🎯 OVERALL ML STATUS: ✅ EXCELLENT
```

**Jika GAGAL:**
- Check model file corruption
- Verify scikit-learn version compatibility
- Check feature extraction errors

---

### Test 3: **TCP Communication**
```bash
# Terminal 1: Start ML Server
python ml_server.py

# Terminal 2: Run TCP Test
python tcp_test_client.py
```

**Expected Output - Server (Terminal 1):**
```
🤖 ML Server Python berjalan di 127.0.0.1:7001
✅ ML Model berhasil dimuat dari model_ml/pickle_model.pkl
🔄 Menunggu koneksi dari Godot ML client...
📞 Client connected: ('127.0.0.1', 12345) - ID: client_1
```

**Expected Output - Client (Terminal 2):**
```
🧪 TEST 1: BASIC CONNECTION
✅ Connected to ML server!
📥 Response received: {"type": "connection_success"}

🧪 TEST 2: ML ETHNIC DETECTION
📤 Sending detection request...
📥 ML Response received:
   - Result: Jawa
   - Confidence: 85.7%
   - ML Mode: real_model  ← PENTING: Harus "real_model"
✅ REAL ML MODEL ACTIVE!

🎯 OVERALL: ✅ ALL TESTS PASSED
```

**Jika GAGAL:**
- Check if ml_server.py is running
- Verify port 7001 is available
- Check firewall settings
- If `ml_mode: "simulation"` → Dependencies belum terinstall dengan benar

---

## 🏗️ Test Integration (Comprehensive) {#test-integration}

### Run Full Integration Test
```bash
python integration_test.py
```

**Expected Output:**
```
🚀 COMPREHENSIVE INTEGRATION TEST SUITE

🔍 PHASE 1: TESTING DEPENDENCIES
🎯 Result: ✅ SUCCESS

🔍 PHASE 2: TESTING ML MODEL DIRECTLY  
🎯 Result: ✅ SUCCESS

🔍 PHASE 3: TESTING TCP COMMUNICATION
🚀 STARTING ML SERVER
✅ ML Server started successfully
🎯 Result: ✅ SUCCESS
🛑 Stopping ML server...

📋 COMPREHENSIVE TEST REPORT
🎯 OVERALL STATUS: 3/3 tests passed

🎉 ALL TESTS PASSED! SISTEM SIAP UNTUK PRODUCTION!
🎯 GODOT INTEGRATION STATUS: READY
```

---

## 🔧 Troubleshooting {#troubleshooting}

### ❌ **Error: No module named 'sklearn'**
```bash
# Solution:
pip install scikit-learn>=1.0.0
```

### ❌ **Error: Model file not found**
```bash
# Check file:
dir model_ml\pickle_model.pkl

# If missing, retrain model using script_training.py
```

### ❌ **Error: Connection refused (TCP)**
```bash
# Check if server is running:
netstat -an | findstr :7001

# Start server:
python ml_server.py
```

### ❌ **Error: ML Mode is 'simulation'**
```bash
# This means dependencies are missing
# Reinstall all dependencies:
pip uninstall scikit-learn scikit-image opencv-python
pip install -r requirements.txt
```

### ❌ **Error: Feature extraction failed**
```bash
# Check image processing libraries:
pip install --upgrade opencv-python Pillow scikit-image
```

### ❌ **Error: Port already in use**
```bash
# Find and kill process using port 7001:
netstat -ano | findstr :7001
taskkill /PID <PID_NUMBER> /F
```

---

## ✅ Checklist Verification {#checklist}

### **Sebelum Test Godot Integration:**

- [ ] **Dependencies Test**: `python test_dependencies.py` → ✅ PASS
- [ ] **ML Model Test**: `python test_ml_model.py` → ✅ PASS  
- [ ] **TCP Test**: `python tcp_test_client.py` → ✅ PASS
- [ ] **Integration Test**: `python integration_test.py` → ✅ PASS
- [ ] **Server Response**: `ml_mode: "real_model"` (NOT "simulation")
- [ ] **Model Predictions**: Konsisten dan confidence > 70%

### **Test dengan Godot:**

1. **Start ML Server**:
   ```bash
   python ml_server.py
   ```
   Wait for: `✅ ML Model berhasil dimuat` + `🔄 Menunggu koneksi`

2. **Run Godot Project**:
   - Open Godot Engine
   - Load project: `tcp-example/project.godot`
   - Run scene dengan ML client

3. **Verify Real ML**:
   - Upload test image di Godot
   - Check server terminal untuk log processing
   - Verify response di Godot UI
   - Confirm: Bukan hasil random/simulasi

### **Expected Godot-Server Communication:**

**Server Log:**
```
📞 Client connected: ('127.0.0.1', xxxxx)
📥 Received: ethnic_detection request
🔍 Starting prediction process...
📊 Ekstraksi fitur selesai: GLCM=20, Color=32, Total=52
🎯 Final Prediction: Jawa (Confidence: 87.3%)
📤 Response sent: real_model prediction
```

**Godot UI:**
```
Ethnicity: Jawa
Confidence: 87.3%
Processing Time: 0.45s
Status: Detection successful
```

---

## 🎯 Final Verification

**✅ Sistem SIAP jika:**
- Semua test scripts mengembalikan SUCCESS
- Server response `ml_mode: "real_model"`
- Godot dapat connect dan receive real predictions
- Predictions konsisten dan masuk akal

**❌ Sistem BELUM SIAP jika:**
- Ada test yang FAIL
- Server response `ml_mode: "simulation"`
- Godot tidak dapat connect
- Predictions random/tidak konsisten

---

## 📞 Support

Jika mengalami masalah:
1. Jalankan `python integration_test.py` untuk comprehensive check
2. Check error logs di terminal
3. Verify semua dependencies dengan `python test_dependencies.py`
4. Pastikan model file `pickle_model.pkl` tidak corrupt
5. Test TCP communication dengan `python tcp_test_client.py`

**Happy Testing! 🎉**