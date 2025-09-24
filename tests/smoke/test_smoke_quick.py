#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def test_env_and_dataset_access():
    assert os.path.exists("dataset")
    assert os.path.exists("dataset/dataset_periorbital")
