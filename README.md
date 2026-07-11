# Biofilm Image Analysis: Segmentation using ADMM & Deep Learning Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Lint & Format](https://github.com/Runtime-Slayers/biofilm-segmentation-admm/actions/workflows/lint.yml/badge.svg)](https://github.com/Runtime-Slayers/biofilm-segmentation-admm/actions)

This repository contains a comprehensive pipeline for the **segmentation, preprocessing, solidification, and classification of biofilm images**. By combining classical mathematical optimization techniques like the **Alternating Direction Method of Multipliers (ADMM)** with state-of-the-art Deep Learning models (EfficientNetB0, U-Net, Mask R-CNN), this project provides a robust workflow for analyzing biofilm datasets.

---

## 🚀 Key Features

### 1. Biofilm Segmentation & Solidification
* **Classical Preprocessing**: Color space conversions, contrast adjustments, and KMeans clustering.
* **Classical Segmentation**: Morphological reconstructions, distance transforms, and Watershed segmentation.
* **Deep Learning Segmentation**: Leverages PyTorch-based **U-Net** (via `segmentation_models_pytorch`) and **Mask R-CNN** for high-precision instance segmentation.
* **ADMM-based Post-Processing**: Applies Alternating Direction Method of Multipliers (ADMM) algorithms for biofilm solidification, enforcing structural and spatial continuity constraints.

### 2. Multi-Class Biofilm Classification
* **Directory Structuring**: Automates sorting of flat validation/test sets into structured folder formats based on inferred class tags from filenames.
* **Imbalance Handling**: Calculates class-weight metrics dynamically to handle class imbalance across 8 categories.
* **Transfer Learning & Data Augmentation**: Utilizes **EfficientNetB0** (pre-trained on ImageNet) combined with rotational, zoom, and contrast-based augmentation layers.
* **Performance Analysis**: Computes classification reports, Macro F1-scores, and generates interactive training histories and confusion matrices.

---

## 📊 Mathematical Formulation of ADMM Solidification

To segment and solidify the biofilm structures, we formulate a multi-objective optimization problem that balances data fidelity, spatial smoothness, and area/volume constraints. Let $x$ represent the primal segmentation mask, $z$ represent the auxiliary constraint variable, and $u$ be the scaled dual variable.

### 1. Objective Function
We minimize the following objective:
$$\min_{x, z} \| x - y \|_2^2 + \mu \operatorname{TV}(x) + \gamma \| x - I_{\text{thresh}} \|_2^2 + \mathbb{I}_{\mathcal{C}}(z)$$

subject to $x = z$, where:
- $y$ is the raw input mask.
- $I_{\text{thresh}}$ represents intensity penalty constraints.
- $\operatorname{TV}(x)$ is the anisotropic Total Variation penalty for spatial smoothness.
- $\mathbb{I}_{\mathcal{C}}(z)$ is the indicator function enforcing area bounds:
$$\mathcal{C} = \left\{ z \;\middle|\; \text{min\_area} \le \sum z \le \text{max\_area}, \;\; 0 \le z_i \le 1 \right\}$$

### 2. ADMM Iterations
The optimization problem is solved iteratively using the Alternating Direction Method of Multipliers:

#### A. Primal $x$-Update
Minimize the augmented Lagrangian with respect to $x$:
$$x^{k+1} = \operatorname{argmin}_{x} \left( \|x - y\|_2^2 + \frac{\rho}{2} \|x - z^k + u^k\|_2^2 + \mu \operatorname{TV}(x) + \gamma \|x - I_{\text{thresh}}\|_2^2 \right)$$

This is computed analytically as:
$$x^{k+1} = \operatorname{clip}\left( \frac{2y + \rho(z^k - u^k) - \lambda - \mu \nabla \operatorname{TV}(x^k) - \text{intensity\_penalty}}{2 + \rho + \gamma}, \;\; 0, \;\; 1 \right)$$

#### B. Primal $z$-Update (Projection)
Project $x^{k+1} + u^k$ onto the convex set $\mathcal{C}$ enforcing area (or volume) bounds:
$$z^{k+1} = \Pi_{\mathcal{C}}(x^{k+1} + u^k)$$

This is solved efficiently by sorting the values and projecting onto the sum constraints.

#### C. Dual $u$-Update
Accumulate the constraint residual:
$$u^{k+1} = u^k + x^{k+1} - z^{k+1}$$

---

## 📂 Repository Structure

```
├── .github/
│   └── workflows/
│       └── lint.yml         # CI workflow for Ruff syntax linting
├── .gitignore             # Git ignore file (excludes datasets, models, manuscripts)
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── setup.py                 # Setuptools installer
├── pyproject.toml           # Configurations for packaging & linting
├── notebooks/             # Step-by-step experiment notebooks
│   ├── biofilm_classifier_efficientnet.ipynb       # Deep Learning Classification workflow
│   └── biofilm_segmentation_and_solidification.ipynb # Preprocessing, Segmentation, and ADMM Solidification
└── src/                     # Core Python modules
    ├── __init__.py          # Exposes package API
    ├── admm_segmentation.py # NumPy implementations of 2D/3D ADMM
    ├── classifier.py        # Model build and weights calculation helpers
    └── dataset.py           # Structuring and sorting dataset directories
```

---

## 🛠️ Installation & Getting Started

### 1. Install Dependencies
You can install this package in editable mode locally:
```bash
pip install -e .
```
This automatically links the `src/` modules so you can import them anywhere on your system.

For deep learning packages like PyTorch and TensorFlow, install them separately based on your GPU capabilities:
```bash
pip install tensorflow>=2.15.0 torch>=2.0.0 torchvision segmentation-models-pytorch
```

### 2. Importing Modules in Python
You can import the core ADMM solvers and dataset helpers directly:
```python
from biofilm_admm import admm_biofilm_segmentation_2d, create_structured_directory

# 1. Structure raw data directories
create_structured_directory("raw_valid/", "valid_structured/", ["control", "vial8 DNA"])

# 2. Run ADMM solidification on a mask slice
solidified_mask, epoch = admm_biofilm_segmentation_2d(
    y=raw_mask, 
    I=intensity_image, 
    min_area=120, 
    max_area=600
)
```

---

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. Open-source contributions are welcome.
