#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np

from tests.helpers.fast_cv_helpers import load_small_sample, extract_features_fast, run_fast_cv

DATASET_DIR = "dataset/dataset_periorbital"

@pytest.mark.skipif(not os.path.exists(DATASET_DIR), reason="dataset not available")
def test_cv_pipeline_small_subset():
    data, labels = load_small_sample(DATASET_DIR, images_per_ethnicity=10)
    assert len(data) > 0 and labels.shape[0] > 0
    features = extract_features_fast(data)
    assert features.shape[0] == labels.shape[0]
    assert features.shape[1] == 52  # 20 GLCM + 32 Color
    scores = run_fast_cv(features, labels)
    assert len(scores) == 6
    assert 0.5 <= np.mean(scores) <= 1.0
