#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP Variants inspired by Smart ICT 2019 (KK_SmartICT2019_Extended.pdf)

Features:
- Multi-scale LBP (multiple radii and sampling points)
- Rotation-invariant options (uniform/riu2 or ror)
- Spatial (block/grid) histograms concatenation
- L1-normalized feature vectors suitable for RF/SVM

LBP Variants implemented:
1. MB-LBP (Multi-Block LBP) - Zhang et al. 2007
2. MBP (Median Binary Pattern) - Hafiane et al.
3. DLBP (Divided LBP) - Hua et al.
4. MQLBP (Multi-quantized LBP) - Patel et al. 2016
5. d-LBP (Doubled LBP) - Two radii neighborhoods
6. RedDLBP (Reduced Divided LBP) - 6 pixels divided into 2 groups

This module exposes various LBP extractors that can be used like other
feature extractors and plugged into the existing pipeline.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from skimage.feature import local_binary_pattern
from scipy import ndimage

from .interfaces import ILogger, IProgressTracker, IFeatureExtractor


class KKPaperLBPExtractor(IFeatureExtractor):
    """Paper-inspired LBP extractor with multi-scale and spatial histograms.

    Expected input: grayscale images array (N, H, W)
    Output: concatenated, L1-normalized histogram features per image.
    """

    def __init__(self,
                 logger: ILogger,
                 progress_tracker: IProgressTracker = None,
                 radii: List[float] = None,
                 points: List[int] = None,
                 method: str = 'uniform',
                 grid_size: Tuple[int, int] = (4, 4)):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.radii = radii or [1, 2, 3]
        self.points = points or [8, 16, 24]
        # method: 'uniform' (riu2-like when binned as P+2) or 'ror'
        self.method = method
        self.grid_size = tuple(grid_size)

        # Precompute bins per scale
        self._bins_per_scale: List[int] = []
        for P in self.points:
            if self.method == 'uniform':
                self._bins_per_scale.append(P + 2)  # riu2 histogram size
            else:
                # For 'ror', use full code space size
                self._bins_per_scale.append(2 ** P)

        self.feature_info = {
            'type': 'LBP_Paper',
            'description': 'Multi-scale, rotation-invariant LBP with spatial histograms',
            'radii': self.radii,
            'points': self.points,
            'method': self.method,
            'grid_size': self.grid_size,
        }

    def _extract_single(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gy, gx = self.grid_size
        cell_h = h // gy
        cell_w = w // gx

        features_all_scales: List[np.ndarray] = []

        for P, R, bins in zip(self.points, self.radii, self._bins_per_scale):
            lbp = local_binary_pattern(img, P, R, method=self.method)

            # Spatial histograms
            hist_parts: List[np.ndarray] = []
            for yi in range(gy):
                for xi in range(gx):
                    y0 = yi * cell_h
                    x0 = xi * cell_w
                    y1 = (yi + 1) * cell_h if yi < gy - 1 else h
                    x1 = (xi + 1) * cell_w if xi < gx - 1 else w
                    region = lbp[y0:y1, x0:x1]

                    # Bin range must match method used
                    if self.method == 'uniform':
                        hist, _ = np.histogram(region.ravel(), bins=bins, range=(0, bins), density=False)
                    else:
                        # ror values are within [0, 2**P)
                        hist, _ = np.histogram(region.ravel(), bins=bins, range=(0, bins), density=False)

                    # L1 normalize
                    s = hist.sum()
                    if s > 0:
                        hist = hist.astype(np.float32) / float(s)
                    else:
                        hist = hist.astype(np.float32)
                    hist_parts.append(hist)

            # Concatenate this scale
            scale_vec = np.concatenate(hist_parts, axis=0).astype(np.float32)
            features_all_scales.append(scale_vec)

        # Concatenate all scales
        feat = np.concatenate(features_all_scales, axis=0)
        return feat

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting Paper-LBP features (multi-scale, spatial histograms)...")
        if self.progress_tracker:
            self.progress_tracker.start_task("Paper-LBP Feature Extraction", len(images))

        feats: List[np.ndarray] = []
        for i, img in enumerate(images):
            feats.append(self._extract_single(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)

        if self.progress_tracker:
            self.progress_tracker.complete_task()

        result = np.vstack(feats)

        # Record dimension and meta
        self.feature_info.update({
            'extracted_count': int(result.shape[0]),
            'feature_dimension': int(result.shape[1]) if result.ndim == 2 else 0,
            'output_shape': list(result.shape)
        })

        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class MBLBPExtractor(IFeatureExtractor):
    """Multi-Block Local Binary Pattern (MB-LBP) - Zhang et al. 2007
    
    Operates on mean values of surrounding blocks rather than individual pixels.
    Uses 2x3 blocks by default, compares mean of surrounding blocks to central block.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 block_size: Tuple[int, int] = (2, 3)):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.block_size = block_size
        
        self.feature_info = {
            'type': 'MB-LBP',
            'description': 'Multi-Block Local Binary Pattern (Zhang et al. 2007)',
            'block_size': self.block_size,
            'feature_dimension': 256  # 8-bit code = 256 possible values
        }
    
    def _extract_mblbp(self, img: np.ndarray) -> np.ndarray:
        """Extract MB-LBP features from single image."""
        h, w = img.shape
        bh, bw = self.block_size
        
        # Ensure image is large enough for blocks
        if h < 3 * bh or w < 3 * bw:
            # Pad image if too small
            pad_h = max(0, 3 * bh - h)
            pad_w = max(0, 3 * bw - w)
            img = np.pad(img, ((pad_h, 0), (pad_w, 0)), mode='reflect')
            h, w = img.shape
        
        # Calculate block means
        block_means = ndimage.uniform_filter(img, size=self.block_size, mode='constant')
        
        # Extract MB-LBP codes
        codes = np.zeros_like(img, dtype=np.uint8)
        
        # 8-neighborhood offsets for central block
        offsets = [(-bh, -bw), (-bh, 0), (-bh, bw), (0, bw), 
                  (bh, bw), (bh, 0), (bh, -bw), (0, -bw)]
        
        for i in range(bh, h - bh):
            for j in range(bw, w - bw):
                center_mean = block_means[i, j]
                code = 0
                
                for bit, (di, dj) in enumerate(offsets):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_mean = block_means[ni, nj]
                        if neighbor_mean > center_mean:
                            code |= (1 << bit)
                
                codes[i, j] = code
        
        # Create histogram
        hist, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256), density=True)
        return hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting MB-LBP features...")
        if self.progress_tracker:
            self.progress_tracker.start_task("MB-LBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_mblbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class MBPExtractor(IFeatureExtractor):
    """Median Binary Pattern (MBP) - Hafiane et al.
    
    Compares each pixel in 3x3 neighborhood with median value of the block.
    Results in 9-bit code including the central pixel.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.feature_info = {
            'type': 'MBP',
            'description': 'Median Binary Pattern (Hafiane et al.)',
            'feature_dimension': 512  # 9-bit code = 512 possible values
        }
    
    def _extract_mbp(self, img: np.ndarray) -> np.ndarray:
        """Extract MBP features from single image."""
        h, w = img.shape
        codes = np.zeros_like(img, dtype=np.uint16)
        
        # 3x3 neighborhood offsets (including center)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), 
                  (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Extract 3x3 neighborhood
                neighborhood = []
                for di, dj in offsets:
                    neighborhood.append(img[i + di, j + dj])
                
                # Calculate median
                median_val = np.median(neighborhood)
                
                # Generate 9-bit code
                code = 0
                for bit, val in enumerate(neighborhood):
                    if val > median_val:
                        code |= (1 << bit)
                
                codes[i, j] = code
        
        # Create histogram
        hist, _ = np.histogram(codes.ravel(), bins=512, range=(0, 512), density=True)
        return hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting MBP features...")
        if self.progress_tracker:
            self.progress_tracker.start_task("MBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_mbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class DLBPExtractor(IFeatureExtractor):
    """Divided Local Binary Pattern (DLBP) - Hua et al.
    
    Divides standard LBP into two parts: even indices and odd indices.
    Each part generates 4-bit code, reducing data range.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.feature_info = {
            'type': 'DLBP',
            'description': 'Divided Local Binary Pattern (Hua et al.)',
            'feature_dimension': 512  # 4-bit + 4-bit = 16 + 16 = 32 bins total
        }
    
    def _extract_dlbp(self, img: np.ndarray) -> np.ndarray:
        """Extract DLBP features from single image."""
        h, w = img.shape
        codes_even = np.zeros_like(img, dtype=np.uint8)
        codes_odd = np.zeros_like(img, dtype=np.uint8)
        
        # 8-neighborhood offsets (even indices: 0,2,4,6; odd indices: 1,3,5,7)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                  (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = img[i, j]
                
                # Even indices (0, 2, 4, 6)
                code_even = 0
                for bit, (di, dj) in enumerate(offsets[::2]):  # Even indices
                    ni, nj = i + di, j + dj
                    if img[ni, nj] > center:
                        code_even |= (1 << bit)
                
                # Odd indices (1, 3, 5, 7)
                code_odd = 0
                for bit, (di, dj) in enumerate(offsets[1::2]):  # Odd indices
                    ni, nj = i + di, j + dj
                    if img[ni, nj] > center:
                        code_odd |= (1 << bit)
                
                codes_even[i, j] = code_even
                codes_odd[i, j] = code_odd
        
        # Create histograms for both parts
        hist_even, _ = np.histogram(codes_even.ravel(), bins=16, range=(0, 16), density=True)
        hist_odd, _ = np.histogram(codes_odd.ravel(), bins=16, range=(0, 16), density=True)
        
        # Concatenate histograms
        combined_hist = np.concatenate([hist_even, hist_odd])
        return combined_hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting DLBP features...")
        if self.progress_tracker:
            self.progress_tracker.start_task("DLBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_dlbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class MQLBPExtractor(IFeatureExtractor):
    """Multi-quantized Local Binary Pattern (MQLBP) - Patel et al. 2016
    
    Extends LTP concept by splitting code into 2L levels.
    Uses both sign and magnitude of differences.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 L: int = 2, thresholds: List[float] = None):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.L = L  # Number of quantization levels
        self.thresholds = thresholds or [0.1, 0.2]  # 2L-1 thresholds
        
        self.feature_info = {
            'type': 'MQLBP',
            'description': f'Multi-quantized LBP (L={L})',
            'L': self.L,
            'thresholds': self.thresholds,
            'feature_dimension': (2 ** (2 * self.L)) * 8  # 2L levels per neighbor, 8 neighbors
        }
    
    def _quantize_difference(self, diff: float) -> int:
        """Quantize difference into 2L levels."""
        if diff < -self.thresholds[-1]:
            return 0
        elif diff < -self.thresholds[0]:
            return 1
        elif diff < 0:
            return 2
        elif diff < self.thresholds[0]:
            return 3
        elif diff < self.thresholds[-1]:
            return 4
        else:
            return 5
    
    def _extract_mqlbp(self, img: np.ndarray) -> np.ndarray:
        """Extract MQLBP features from single image."""
        h, w = img.shape
        codes = np.zeros_like(img, dtype=np.uint16)
        
        # 8-neighborhood offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                  (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = img[i, j]
                code = 0
                
                for bit, (di, dj) in enumerate(offsets):
                    ni, nj = i + di, j + dj
                    diff = (img[ni, nj] - center) / 255.0  # Normalize
                    quantized = self._quantize_difference(diff)
                    code |= (quantized << (bit * 3))  # 3 bits per neighbor
                
                codes[i, j] = code
        
        # Create histogram
        max_val = (2 ** (3 * 8)) - 1  # Maximum possible code
        hist, _ = np.histogram(codes.ravel(), bins=min(1024, max_val + 1), 
                              range=(0, min(1024, max_val + 1)), density=True)
        return hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info(f"Extracting MQLBP features (L={self.L})...")
        if self.progress_tracker:
            self.progress_tracker.start_task("MQLBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_mqlbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class DLBPExtractor(IFeatureExtractor):
    """Doubled Local Binary Pattern (d-LBP)
    
    Uses two neighborhoods: 8 pixels at radius 1 and 8 pixels at radius 3.
    Results in two LBP codes and two local histograms.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 radius1: int = 1, radius2: int = 3, n_points: int = 8):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.radius1 = radius1
        self.radius2 = radius2
        self.n_points = n_points
        
        self.feature_info = {
            'type': 'd-LBP',
            'description': f'Doubled LBP (r1={radius1}, r2={radius2})',
            'radius1': self.radius1,
            'radius2': self.radius2,
            'n_points': self.n_points,
            'feature_dimension': 512  # 256 + 256 for two histograms
        }
    
    def _extract_dlbp(self, img: np.ndarray) -> np.ndarray:
        """Extract d-LBP features from single image."""
        # Extract LBP for radius 1
        lbp1 = local_binary_pattern(img, self.n_points, self.radius1, method='uniform')
        
        # Extract LBP for radius 3
        lbp2 = local_binary_pattern(img, self.n_points, self.radius2, method='uniform')
        
        # Create histograms
        hist1, _ = np.histogram(lbp1.ravel(), bins=256, range=(0, 256), density=True)
        hist2, _ = np.histogram(lbp2.ravel(), bins=256, range=(0, 256), density=True)
        
        # Concatenate histograms
        combined_hist = np.concatenate([hist1, hist2])
        return combined_hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info(f"Extracting d-LBP features (r1={self.radius1}, r2={self.radius2})...")
        if self.progress_tracker:
            self.progress_tracker.start_task("d-LBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_dlbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class RedDLBPExtractor(IFeatureExtractor):
    """Reduced Divided Local Binary Pattern (RedDLBP)
    
    Uses 6 pixels at radius 2, divided into two groups of 3 pixels each.
    Results in two 3-bit codes concatenated to form descriptor.
    """
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 radius: int = 2):
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.radius = radius
        
        # 6-pixel neighborhood at radius 2 (divided into 2 groups of 3)
        self.group1_offsets = [(-2, 0), (0, -2), (2, 0)]  # Top, left, bottom
        self.group2_offsets = [(0, 2), (-2, 2), (2, -2)]  # Right, top-right, bottom-left
        
        self.feature_info = {
            'type': 'RedDLBP',
            'description': f'Reduced Divided LBP (r={radius})',
            'radius': self.radius,
            'feature_dimension': 128  # 8 + 8 for two 3-bit codes
        }
    
    def _extract_reddlbp(self, img: np.ndarray) -> np.ndarray:
        """Extract RedDLBP features from single image."""
        h, w = img.shape
        codes_group1 = np.zeros_like(img, dtype=np.uint8)
        codes_group2 = np.zeros_like(img, dtype=np.uint8)
        
        for i in range(self.radius, h - self.radius):
            for j in range(self.radius, w - self.radius):
                center = img[i, j]
                
                # Group 1: 3 pixels
                code1 = 0
                for bit, (di, dj) in enumerate(self.group1_offsets):
                    ni, nj = i + di, j + dj
                    if img[ni, nj] > center:
                        code1 |= (1 << bit)
                
                # Group 2: 3 pixels
                code2 = 0
                for bit, (di, dj) in enumerate(self.group2_offsets):
                    ni, nj = i + di, j + dj
                    if img[ni, nj] > center:
                        code2 |= (1 << bit)
                
                codes_group1[i, j] = code1
                codes_group2[i, j] = code2
        
        # Create histograms for both groups
        hist1, _ = np.histogram(codes_group1.ravel(), bins=8, range=(0, 8), density=True)
        hist2, _ = np.histogram(codes_group2.ravel(), bins=8, range=(0, 8), density=True)
        
        # Concatenate histograms
        combined_hist = np.concatenate([hist1, hist2])
        return combined_hist.astype(np.float32)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info(f"Extracting RedDLBP features (r={self.radius})...")
        if self.progress_tracker:
            self.progress_tracker.start_task("RedDLBP Feature Extraction", len(images))
        
        features = []
        for i, img in enumerate(images):
            features.append(self._extract_reddlbp(img))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'output_shape': list(result.shape)
        })
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


