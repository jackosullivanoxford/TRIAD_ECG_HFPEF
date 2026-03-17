# TRIAD_ECG_HFPEF

**Multimodal Machine Learning for the Genomic and Proteomic Architecture of Heart Failure with Preserved Ejection Fraction (HFpEF).**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

This repository contains the official implementation of the **TRIAD-ECG** model, a deep learning system designed to predict HFpEF status and associated phenotypes from 12-lead electrocardiograms.

The model utilizes a custom CNN architecture with integrated **spatial and channel attention** (the *jos architecture*) to identify subtle morphological features in ECG signals that correlate with the **multiomic architecture of HFpEF**.

---

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/jackosullivan/TRIAD-ECG.git
cd TRIAD-ECG
pip install -r requirements.txt
```

### 2. Generate Demo Data

Since clinical ECG data is protected, we provide a simulation script to generate a **synthetic cohort of 100 records** for testing the pipeline.

```bash
python scripts/generate_demo_data.py
```

### 3. Train Model

```bash
python -c "import ecg; ecg.ecg(task='HFpEF')"
```

---

## Architecture & Specs

**Model:** Custom CNN with Spatial/Channel Attention (*jos architecture*)

**Parameters:**  
- ~1M+ trainable parameters

**Input:**  
- 12-lead ECG  
- 5,000 samples @ 500 Hz  
- Downsampled 2× → 2,500 samples

**Training:**  
- Optimizer: AdamW (`lr = 1e-5`)  
- Scheduler: `ReduceLROnPlateau`  
- Loss: Binary cross-entropy with class weighting

---

## Repository Structure

```text
TRIAD_ECG_HFPEF/
├── ecg.py                     # Main interface for training and evaluation
├── model.py                   # Core ECG model with attention layers
├── model_specs.py             # Architecture specifications (jos)
├── dataset.py                 # ECG data loading and preprocessing
├── config.py                  # Hyperparameters and task definitions
├── train.py                   # Training loop logic
├── scripts/
│   └── generate_demo_data.py  # Synthetic data generator
├── data/                      # Local directory for waveforms and labels
├── docs/                      # Technical documentation
└── requirements.txt           # Dependencies
```

---

## Usage

### Advanced Training

You can override any configuration parameter via the `cfg_updates` dictionary:

```python
import ecg

ecg.ecg(
    task='HFpEF',
    cfg_updates={
        'optimizer': {
            'lr': 1e-4,
            'batch_size': 64
        }
    }
)
```

### Evaluation

To evaluate a pre-trained model on a test split:

```python
import ecg

ecg.ecg(
    mode='eval',
    task='HFpEF',
    eval_model_path='checkpoints/best_model.pt'
)
```

---

## Citation

If you use this code or the **TRIAD-ECG architecture** in your research, please cite:

```
O'Sullivan JW, et al.
Multimodal Machine Learning Reveals the Genomic and Proteomic
Architecture of Heart Failure with Preserved Ejection Fraction.
https://www.medrxiv.org/content/10.64898/2026.02.07.26345811v2
```

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.  
It is intended for **academic and research use only**.

---

## Contact

**Jack W O'Sullivan**  
Stanford University  

jackos@stanford.edu

pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
tqdm>=4.60.0
torchinfo>=1.5.0
