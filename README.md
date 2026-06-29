# Automated EdTech Grading Assistant

This repository contains the source code, notebooks, and reports for the **Automated EdTech Grading Assistant** project, focusing on handwritten digit image classification using classical machine learning pipelines.

## Repository Structure

```
EdTech-Grading-Assistant/
├── notebooks/
│   └── 01_data_loading.ipynb     # Data ingestion, reshaping math, and visualization EDA
├── src/
│   └── loader.py                 # Modular, cached, and SSL-bypassed MNIST data loader
├── images/                       # Folder for storing plots, visual grids, and figures
├── reports/
│   └── MNIST_VISUALIZATION_REPORT.md # Technical report detailing ingestion and spatial visualization
└── README.md                     # Project documentation overview (this file)
```

## Quick Start & Usage

### Prerequisites
Make sure Python 3.10+ is installed, along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 1. Programmatic Data Loading
The `src/loader.py` module provides a reusable function to load the MNIST dataset from OpenML. It returns features and target labels as standard NumPy arrays:
```python
from src.loader import load_mnist

# Load dataset (70,000 samples, 784 features per sample)
X, y = load_mnist(version=1, as_frame=False)
```

### 2. Exploring Visualizations
Open the Jupyter notebook inside `notebooks/01_data_loading.ipynb` to inspect:
* Data dimensions and class distribution balancing.
* Reshaping 1D pixel vectors ($784$ features) to 2D image matrices ($28 \times 28$ pixels).
* Single-digit detailed matrix cell plots mapping out exact integer pixel values (0–255).
* Random sample grids and representative class grids for numbers 0–9.
* Pixel intensity distributions pointing out structural sparsity and scaling considerations.

---

## Technical Concept Notes

* **Reshaping**: Mapping index $i$ in a 1D vector to row $r$ and column $c$ on a $28 \times 28$ grid:
  $$i = r \times 28 + c$$
* **Sparsity**: Over 80% of MNIST pixels have an intensity of 0 (background), which is highly beneficial for SVM hyperplanes.
* **Feature Normalization**: Pixel normalization to `[0, 1]` or standardizing to unit variance is required prior to running RBF kernel SVMs to prevent scale saturation.