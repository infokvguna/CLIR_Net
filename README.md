# CLIR-Net: Cross-Lingual Invariant Representation Network for Speech and Audio-Visual Emotion Recognition

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-CLIR--Net-purple.svg)](#citation)

Official PyTorch implementation of **CLIR-Net** (Cross-Lingual Invariant Representation Network), a causal framework designed for robust cross-lingual and multimodal emotion recognition.

---

## 📌 Overview

Cross-lingual emotion recognition faces severe performance degradation due to language-specific acoustic variations, speaker characteristics, and cultural expressions. **CLIR-Net** addresses this challenge by formulating emotion recognition through a **Structural Causal Model (SCM)** that explicitly disentangles invariant emotion features from language and speaker confounders.

### Key Innovations:
1. **Multimodal Encoders**:
   - **Acoustic**: Pretrained Wav2Vec 2.0 / XLS-R representations fused with frame-level prosodic contours (pitch $F_0$, RMS energy, spectral centroid, spectral rolloff, spectral flux).
   - **Visual**: Temporal Transformer encoder modeling dynamics across 16 Facial Action Units (FAUs) extracted via MediaPipe.
2. **Structural Causal Model (SCM)**:
   - Learnable continuous gates estimating causal dependencies between Language ($L$), Speaker ($S$), Acoustics ($A$), Facial Dynamics ($F$), Voice Quality ($VQ$), and Latent Emotion ($E$).
   - NOTEARS-based matrix exponential acyclicity penalty enforcing Directed Acyclic Graph (DAG) constraints.
3. **Counterfactual Invariance Regularization**:
   - Synthesizes counterfactual representations under intervention $\text{do}(L = L')$.
   - Enforces $L_2$ invariance loss ($L_\text{inv}$) and symmetric KL divergence prediction consistency loss ($L_\text{cf\_pred}$) so latent emotion $E$ remains stable across linguistic shifts.

---

## 🏗️ Architecture & Causal Formulation

```
                     ┌──────────────────┐
                     │  Speaker (S)     │──────────────┐
                     └─────────┬────────┘              │
                               │                       ▼
                               │               ┌───────────────┐
                               ▼         ┌────►│Voice Qual (VQ)│
┌──────────────────┐     ┌───────────┐   │     └───────┬───────┘
│  Language (L)    ├────►│Acoustic(A)├───┘             │
└────────┬─────────┘     └─────┬─────┘                 │
         │                     │                       │
         ▼                     ▼                       ▼
   ┌───────────┐         ┌───────────────────────────────┐
   │Facial (F) ├────────►│     Latent Emotion (E)        │
   └───────────┘         └──────────────┬────────────────┘
                                        │
                                        ▼
                                 [Emotion Head]
                                        │
                                        ▼
                             Predicted Emotion Class
```

### Loss Formulation
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{cls} + \alpha \mathcal{L}_\text{inv} + \beta \mathcal{L}_\text{cf\_pred} + \gamma \mathcal{L}_\text{acyc}$$

- **$\mathcal{L}_\text{cls}$**: Label-smoothed Cross-Entropy classification loss.
- **$\mathcal{L}_\text{inv} = \|E - E_{\text{do}(L=L')}\|^2_2$**: Invariance of latent emotion under counterfactual language intervention.
- **$\mathcal{L}_\text{cf\_pred} = D_\text{KL}(p(y|E) \parallel p(y|E_{\text{do}(L=L')}))$$: Prediction consistency across interventions.
- **$\mathcal{L}_\text{acyc} = \text{tr}(e^{W \odot W}) - d$**: Continuous NOTEARS DAG penalty.

---

## 📁 Repository Structure

```
CLIR-Net/
├── models/
│   ├── clirnet.py            # Complete CLIR-Net model integrating all submodules
│   ├── scm.py                # Structural Causal Model with learnable edge gates
│   ├── audio_encoder.py      # Wav2Vec 2.0 / XLS-R + prosodic temporal fusion
│   ├── au_encoder.py         # 16-Action Unit Transformer encoder
│   ├── emotion_head.py       # Linear emotion classification head
│   └── __init__.py
├── data_loaders/
│   ├── base_dataset.py       # EmotionDataset & MultiCorpusDataset loaders
│   ├── download.py           # Dataset acquisition and directory setup utilities
│   └── __init__.py
├── preprocessing/
│   ├── audio_features.py     # Waveform, Mel, F0, energy, jitter, shimmer, HNR
│   ├── video_features.py     # MediaPipe FaceMesh 16 Action Unit extraction
│   └── __init__.py
├── scripts/
│   ├── train_clirnet.py      # Primary training script with multi-GPU/CPU support
│   ├── train.py              # Shortcut training entrypoint
│   ├── evaluate_model.py     # Evaluation, metrics generation, and confusion matrix
│   ├── preprocess_dataset.py # End-to-end dataset preprocessing pipeline
│   ├── test_all.py           # Comprehensive sanity check and unit test suite
│   └── __init__.py
├── training/
│   ├── losses.py             # Classification, Invariance, CF, and Acyclicity losses
│   └── __init__.py
├── config.py                 # Central hyperparameter and path dataclass configuration
├── requirements.txt          # Python dependency specifications
├── setup.py                  # Standard packaging setup script
├── pyproject.toml            # PEP 518/621 build configuration
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/<your-username>/CLIR-Net.git
cd CLIR-Net

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CLIR-Net in editable mode
pip install -e .
```

### 2. Verify Installation

Run the complete component test suite to verify CUDA, model dimensions, and data pipelines:

```bash
python scripts/test_all.py
```

---

## 📊 Datasets & Preprocessing

CLIR-Net supports five standard multilingual emotion recognition corpora:

| Dataset | Language | Modality | # Emotions | Access / Source |
| :--- | :--- | :--- | :---: | :--- |
| **EMO-DB** | German | Audio | 6 | [Bilderbar](http://emodb.bilderbar.info/download/) (Auto-downloadable) |
| **EMOVO** | Italian | Audio | 6 | [GitHub / EmoV-DB](https://github.com/numediart/EmoV-DB) |
| **ShEMO** | Persian | Audio | 6 | [GitHub / ShEMO](https://github.com/pariajm/sharif-emotional-speech-database) |
| **CREMA-D** | English | Audio-Visual | 6 | [GitHub / CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) |

### Automated Dataset Setup

Check dataset status and trigger downloads:

```bash
# Check available datasets
python data_loaders/download.py --check

# Display download and setup instructions
python data_loaders/download.py --instructions

# Automatically download public datasets (e.g., EMO-DB)
python data_loaders/download.py --setup --data_root "./data"
```

### Feature Preprocessing

Preprocess raw audio waveforms and extract frame-level prosodic and Action Unit features:

```bash
# Preprocess single corpus (e.g., EMO-DB)
python scripts/preprocess_dataset.py --dataset emodb --data_root "./data"

# Preprocess CREMA-D (Audio + Action Units)
python scripts/preprocess_dataset.py --dataset cremad --data_root "./data"

# Preprocess all available corpora
python scripts/preprocess_dataset.py --dataset all --data_root "./data"
```

Each preprocessed corpus produces `train_metadata.json`, `val_metadata.json`, `test_metadata.json`, and precomputed `.pt` tensors under `data/processed/<corpus>/`.

---

## ⚡ Training

### 1. Cross-Lingual Pooled Training
Train across all multilingual datasets simultaneously with counterfactual causal regularization:

```bash
python scripts/train_clirnet.py \
    --pooled \
    --data_dir ./data/processed/emodb \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --checkpoint_dir ./checkpoints
```

### 2. Single-Language Training
Train on a target corpus (e.g., German EMO-DB):

```bash
python scripts/train_clirnet.py \
    --data_dir ./data/processed/emodb \
    --language german \
    --batch_size 16 \
    --num_epochs 50 \
    --lr 1e-4
```

### 3. Audio-Only Ablation
Train the model without video/AU inputs:

```bash
python scripts/train_clirnet.py \
    --audio_only \
    --data_dir ./data/processed/emodb \
    --language german \
    --batch_size 16
```

---

## 📈 Evaluation

Evaluate trained checkpoints on held-out test splits to generate classification metrics and plots:

```bash
python scripts/evaluate_model.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/processed/emodb \
    --language german \
    --output_dir ./results
```

The script outputs:
- **`test_metrics.json`**: Overall Accuracy, Weighted F1, Macro F1, Precision, Recall, and per-class breakdowns.
- **`confusion_matrix.png`**: High-resolution annotated confusion matrix.
- **`per_class_f1.png`**: Per-class F1 score bar charts.

---

## ⚙️ Configuration (`config.py`)

All hyperparameters can be centrally configured in [config.py](config.py):

```python
from config import get_config

cfg = get_config()
cfg.model.audio_pretrained = "facebook/wav2vec2-base"  # or "facebook/wav2vec2-xls-r-300m"
cfg.training.alpha_inv = 1.0       # Invariance loss weight
cfg.training.beta_cf_pred = 1.0    # Counterfactual prediction consistency weight
cfg.training.gamma_acyc = 1e-2     # DAG acyclicity weight
```

---

## 📑 Citation

If you use **CLIR-Net** in your research, please cite our paper:

```bibtex
@article{clirnet2026,
  title     = {CLIR-Net: Cross-Lingual Invariant Representation Network for Speech and Audio-Visual Emotion Recognition via Causal Intervention},
  author    = {CLIR-Net Research Team},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
