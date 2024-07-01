# Fingerprint Matcher

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.5.2-informational)
![License](https://img.shields.io/badge/license-MIT-green)

Python application for fingerprint matching using SIFT and FLANN algorithms.

## Overview

The Fingerprint Matcher project is designed to match a test fingerprint image against multiple databases stored in separate folders within a common directory. It utilizes computer vision techniques provided by OpenCV for feature detection and matching.

## Features

- **Key Features:**
  - **SIFT Algorithm:** Detects keypoints and computes descriptors for fingerprint images.
  - **FLANN Matcher:** Utilizes FLANN (Fast Library for Approximate Nearest Neighbors) to find best matches between keypoints.
  - **Database Search:** Iterates through multiple database folders to find fingerprint matches.

## Usage

1. **Install Dependencies:**
   ```bash
   pip install opencv-python-headless numpy
