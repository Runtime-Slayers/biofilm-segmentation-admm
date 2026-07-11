# Biofilm Image Analysis: Segmentation using ADMM & Deep Learning Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

This repository contains a comprehensive pipeline for the **segmentation, preprocessing, solidification, and classification of biofilm images**. By combining classical mathematical optimization techniques like the **Alternating Direction Method of Multipliers (ADMM)** with state-of-the-art Deep Learning models (EfficientNetB0, U-Net, Mask R-CNN), this project provides a robust workflow for analyzing biofilm datasets.

---

## 🚀 Key Features

### 1. Biofilm Segmentation & Solidification
Located in `notebooks/biofilm_segmentation_and_solidification.ipynb`, this workflow includes:
* **Image Preprocessing**: Color space conversions, contrast adjustments, and KMeans clustering.
* **Classical Segmentation**: Morphological reconstructions, distance transforms, and Watershed segmentation.
* **Deep Learning Segmentation**: Leverages PyTorch-based **U-Net** (via `segmentation_models_pytorch`) and **Mask R-CNN** for high-precision instance segmentation.
* **ADMM-based Post-Processing**: Applies Alternating Direction Method of Multipliers (ADMM) algorithms for biofilm solidification, enforcing structural and spatial continuity constraints.

### 2. Multi-Class Biofilm Classification
Located in `notebooks/biofilm_classifier_efficientnet.ipynb`, this workflow includes:
* **Directory Structuring**: Automates sorting of flat validation/test sets into structured folder formats based on inferred class tags from filenames.
* **Imbalance Handling**: Calculates class-weight metrics dynamically to handle class imbalance across 8 categories (e.g., `NC1`, `NC6`, `NC_2`, `NC_3`, `NC_4`, `NC_5`, `control`, `vial8 DNA`).
* **Transfer Learning & Data Augmentation**: Utilizes **EfficientNetB0** (pre-trained on ImageNet) combined with rotational, zoom, and contrast-based augmentation layers.
* **Performance Analysis**: Computes classification reports, Macro F1-scores, and generates interactive training histories and confusion matrices.

---

## 📂 Repository Structure

```
├── .gitignore             # Files and directories ignored by Git (datasets, manuscripts, etc.)
├── LICENSE                # MIT License
├── README.md              # Project overview and guide
├── requirements.txt       # List of Python dependencies
└── notebooks/             # Executable Jupyter Notebooks
    ├── biofilm_classifier_efficientnet.ipynb       # Deep Learning Classification workflow
    └── biofilm_segmentation_and_solidification.ipynb # Preprocessing, Segmentation, and ADMM Solidification
```

---

## 🛠️ Prerequisites & Installation

To run these notebooks locally or in your own workspace, you will need **Python 3.11+** installed.

### 1. Install Dependencies
Install all package prerequisites via `pip`:
```bash
pip install -r requirements.txt
```

*Note: For GPU acceleration, ensure your CUDA/cuDNN drivers are configured and match your TensorFlow/PyTorch installations.*

### 2. Primary Libraries Used
* **Deep Learning**: `TensorFlow`, `Keras`, `PyTorch` (`torch`, `torchvision`), `segmentation-models-pytorch`
* **Computer Vision & Image Processing**: `OpenCV (cv2)`, `scikit-image (skimage)`, `scipy`
* **Data Science & Visualization**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
* **Utilities**: `tqdm`, `glob`

---

## 📖 Usage Instructions

### Run on Kaggle or Google Colab
* The notebooks are pre-configured to handle both Kaggle environment paths (`/kaggle/input/...`) and Google Colab upload widgets. 
* To run the classifier on Kaggle, upload your image zip archive as a dataset named `biofilm-dataset` or `biofilm-train-labelled`, and update the `INPUT_DIR` configuration in the notebook setup cell.

### Run Locally
1. Download or extract your dataset into a local folder (e.g., `./data`).
2. Update the input paths inside the notebooks to point to your local dataset path:
   ```python
   INPUT_DIR = "./data/biofilm_train_labelled"
   ```
3. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Run the code cells sequentially.

---

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. Contributions are freely available, and open-source contributions are welcome.
