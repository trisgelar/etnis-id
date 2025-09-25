# 🧪 Testing Guide

This project uses a pipeline-based test structure powered by `pytest`.

## Contents
1. Overview
2. Setup
3. Running Tests
4. Suites and Expectations
5. Troubleshooting
6. Verification Checklist

---

## 1) Overview

The test suite verifies:
- Dependencies and environment
- Feature extractors (GLCM, Color Histogram)
- Cross-validation pipeline (subset of real dataset)
- Configuration system via `.env`
- SOLID training orchestration
- Visualization generation (confusion matrix, feature importance)

---

## 2) Setup (Windows)
```
# From the project root
cmd /c "env\Scripts\activate.bat && pip install -r requirements.txt && pip install pytest"
```

Dataset location used by tests:
```
dataset/dataset_periorbital
```
If missing, some tests will be skipped.

---

## 3) Running Tests

- All tests:
```
cmd /c "env\Scripts\activate.bat && python -m pytest -q"
```

- By suite:
```
# Unit (fast)
cmd /c "env\Scripts\activate.bat && pytest -q tests/unit"

# Cross-validation pipeline
cmd /c "env\Scripts\activate.bat && pytest -q tests/cv"

# Configuration system
cmd /c "env\Scripts\activate.bat && pytest -q tests/config"

# SOLID training
cmd /c "env\Scripts\activate.bat && pytest -q tests/solid"

# Visualization
cmd /c "env\Scripts\activate.bat && pytest -q tests/visualization"

# Smoke (environment/dataset presence)
cmd /c "env\Scripts\activate.bat && pytest -q tests/smoke"
```

---

## 4) Suites and Expectations

- tests/unit
  - Validates feature dimensions: GLCM=20, Color=32
  - Should run in seconds

- tests/cv
  - Uses a small subset (e.g., 10–30 images per class)
  - Verifies 52-dim combined features and successful 6-fold CV scoring

- tests/config
  - Loads `.env` and validates typed accessors

- tests/solid
  - Orchestrates components; may take longer

- tests/visualization
  - Generates figures under `logs/`; can be skipped in CI if needed

- tests/smoke
  - Quick checks for dataset presence and environment readiness

---

## 5) Troubleshooting

- No module named pytest
```
cmd /c "env\Scripts\activate.bat && pip install pytest"
```

- Dataset-related tests failing or skipped
```
Check dataset/dataset_periorbital exists and contains class folders
```

- scikit-image GLCM import errors
```
from skimage.feature import graycomatrix, graycoprops
# If older version:
from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
```

- Matplotlib font warnings for emojis
```
Safe to ignore; they do not affect test correctness
```

---

## 6) Verification Checklist

- [ ] `pytest` runs without import errors
- [ ] Unit tests validate feature dimensions (20 + 32)
- [ ] CV tests produce 6 scores and 52-dim features
- [ ] Config tests load values from `.env`
- [ ] Smoke tests confirm dataset presence
- [ ] Visualization tests save images under `logs/`

Happy testing! 🎉