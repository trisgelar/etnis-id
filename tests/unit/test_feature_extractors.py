#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pytest

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from skimage.measure import shannon_entropy


def create_synthetic_image(width=32, height=32, color=(128, 200, 50)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_glcm_feature_length():
    img = create_synthetic_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3/4*np.pi],
                        symmetric=True, normed=True, levels=256)
    props = ['contrast', 'homogeneity', 'correlation', 'ASM']
    feats = np.hstack([graycoprops(glcm, p).ravel() for p in props])
    entropy = [shannon_entropy(glcm[:, :, :, idx]) for idx in range(glcm.shape[3])]
    feat = np.concatenate((entropy, feats), axis=0)
    assert feat.shape[0] == 20


def test_color_histogram_feature_length():
    img = create_synthetic_image()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist2 = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    arr = np.concatenate((hist1, hist2)).flatten()
    assert arr.shape[0] == 32
