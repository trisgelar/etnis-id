#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
from skimage.measure import shannon_entropy


def load_small_sample(data_dir, images_per_ethnicity=20):
    X, y = [], []
    ethnicities = os.listdir(data_dir)
    for ethnicity in ethnicities:
        p = os.path.join(data_dir, ethnicity)
        if not os.path.isdir(p):
            continue
        files = [f for f in os.listdir(p) if f.lower().endswith('.jpg')][:images_per_ethnicity]
        for img_file in files:
            img_path = os.path.join(p, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                X.append(image)
                y.append(ethnicity)
    return X, np.array(y)


def extract_features_fast(data):
    glcm_features = []
    color_features = []
    for image in data:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
        try:
            glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True, levels=256)
            properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
            feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
            entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
            glcm_feat = np.concatenate((entropy, feats), axis=0)
            glcm_features.append(glcm_feat)
        except Exception:
            glcm_features.append(np.zeros(20))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist2 = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        color_feat = np.concatenate((hist1, hist2)).flatten()
        color_features.append(color_feat)
    return np.concatenate((np.array(glcm_features), np.array(color_features)), axis=1)


def run_fast_cv(features, labels):
    X, y = shuffle(features, labels, random_state=220)
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    cv = StratifiedKFold(n_splits=6)
    return cross_val_score(clf, X, y, cv=cv)
