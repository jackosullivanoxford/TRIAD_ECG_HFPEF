# TRIAD_ECG_HFPEF
This is the model code to train, validate and test a model to predict HFpEF from 12 lead ECGs

## Quick Start
```bash
pip install -r requirements.txt
python -c "import ecg; ecg.ecg(task='HFpEF')"
```

## Architecture
- **Model**: Custom CNN with spatial/channel attention (jos architecture)
- **Input**: 12-lead ECG (5000 samples @ 500Hz, downsampled 2x)
- **Tasks**: 15+ HFpEF phenotypes (ICD codes, subclinical, biomarkers)

## Key Files
- `model.py` - Core ECG model with attention layers
- `model_specs.py` - Architecture specifications (jos)
- `train.py` - Training loop & evaluation
- `dataset.py` - ECG data loading & preprocessing
- `config.py` - Hyperparameters & task definitions
- `ecg.py` - Main training/eval interface

## Usage
```python
import ecg
# Train model
ecg.ecg(task='HFpEF', cfg_updates={'optimizer': {'lr': 1e-4}})
# Evaluate
ecg.ecg(mode='eval', task='HFpEF', eval_model_path='/path/to/model')
```

## Model Specs
- 1M+ parameters
- Batch size: 32
- AdamW optimizer (lr=1e-5)
- ReduceLROnPlateau scheduler
- Binary cross-entropy loss with class weighting
```

### 2. **Repository Structure**
```
ai-ecg-hfpef/
├── README.md
├── requirements.txt
├── setup.py (optional)
├── config.py
├── ecg.py
├── model.py
├── model_specs.py
├── dataset.py
├── train.py
├── examples/
│   ├── train_example.py
│   └── eval_example.sh
└── docs/
    ├── architecture.md (optional: model diagrams)
    └── tasks.md (optional: task descriptions)
```

### 3. **requirements.txt**
```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
tqdm>=4.60.0
torchinfo>=1.5.0
