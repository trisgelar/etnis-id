#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

try:
    print("Testing configuration system...")
    from ml_training.core.config import get_config
    config = get_config()
    print(f"Dataset path: {config.dataset.periorbital_dir}")
    print(f"Ethnicities: {config.dataset.ethnicities}")
    print("Configuration loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

