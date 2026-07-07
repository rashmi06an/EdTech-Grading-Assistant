# 🎓 Automated EdTech Grading Assistant

This repository contains the complete production-grade source code, Jupyter notebooks, interactive web server application, and technical reports for the **Automated EdTech Grading Assistant** (Handwritten Image Classification) project. 

This project implements a classical machine learning pipeline using **Support Vector Machines (SVM)** with **Radial Basis Function (RBF) kernels** to classify handwritten digits (0–9) from the MNIST database, achieving a classification accuracy of **97.38%** (exceeding the target benchmark of 97%).

---

## 📊 Key Project Benchmarks

Through rigorous hyperparameter grid sweeps and model comparisons, the following performance metrics were established:

| Model | Accuracy | Macro F1-Score | Training Time (s) | Inference Latency (s / 5k samples) | CPU Deployment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimized RBF SVM** ($C=10$, $\gamma=\text{'scale'}$) | **97.38%** | **97.35%** | 103.64s | 7.22s (~1.4 ms/digit) | **Highly Feasible** |
| **Random Forest** (100 Trees) | 95.82% | 95.78% | **1.27s** | **0.026s (~5 μs/digit)** | **Excellent** |
| **Baseline Linear SVM** ($C=1.0$) | 92.10% | 91.97% | 15.39s | 15.39s | **Feasible** |

> [!TIP]
> **RBF SVM** provides the highest grading accuracy, making it ideal for high-stakes examinations. However, **Random Forest** trains nearly **80x faster** and classifies digits in **microseconds**, making it incredibly efficient for low-resource environments.

---

## 📁 Repository Structure

```
EdTech-Grading-Assistant/
├── notebooks/
│   ├── 01_data_loading.ipynb              # EDA, pixel spatial grids, and annotated single-digit plots
│   ├── 02_model_training_and_evaluation.ipynb # Master validation notebook (pre-run with grid sweeps & heatmaps)
│   └── generate_notebook.py               # Programmatic notebook generator script
├── src/
│   ├── loader.py                          # SSL-bypassed MNIST dataset loader with local caching
│   ├── scaler.py                          # Preprocessing pixel scaling (custom division / MinMaxScaler) & splitting
│   ├── svm_classifier.py                  # SVM classifier module (baseline training, GridSearchCV, CM heatmap)
│   ├── random_forest.py                   # RandomForest model module (speed and benchmarking)
│   ├── train.py                           # Master training script for model training and comparison log compilation
│   ├── inference.py                       # CLI inference helper (supports random demos and custom PNG drawing files)
│   ├── server.py                          # Lightweight backend HTTP API server
│   └── index.html                         # Premium drawing canvas frontend with real-time processed preview
├── models/
│   └── best_svm_model.joblib              # Saved trained SVM RBF model (generated after training)
├── images/
│   └── confusion_matrix.png               # Saved test set evaluation confusion matrix heatmap
├── reports/
│   ├── grid_search_log.csv                # Tabular log of hyperparameter sweep mean accuracies
│   ├── model_comparison.csv               # Tabular log of side-by-side benchmark metrics
│   ├── MNIST_VISUALIZATION_REPORT.md      # Report detailing data ingestion and spatial characteristics
│   ├── RUN_MANUAL.md                      # Detailed step-by-step setup and running manual
│   └── TECHNICAL_REPORT.md                # Comprehensive technical report on SVM theory, tuning, and errors
├── .gitignore                             # Untracks large binaries and presentation slides from GitHub
└── README.md                              # Repository overview (this document)
```

---

## 🛠️ Quick Start & Usage

Ensure you have Python 3.10+ installed.

### 1. Installation
Install core scientific computing and visualization dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn Pillow
```

### 2. Model Training & Evaluation
To load MNIST, normalize pixel ranges, perform cross-validated hyperparameter tuning (GridSearchCV), and save the final optimized SVM model:
```bash
python3 src/train.py
```
* **Inputs:** Raw MNIST database downloaded and cached locally.
* **Outputs:** 
  * Model saved to `models/best_svm_model.joblib`.
  * Confusion matrix heatmap saved to `images/confusion_matrix.png`.
  * Grid search log saved to `reports/grid_search_log.csv`.
  * Benchmark table saved to `reports/model_comparison.csv`.

### 3. Command-Line Inference (Predictions)
Run a demo prediction on a random MNIST sample (represented as an ASCII art diagram in your terminal):
```bash
python3 src/inference.py --demo
```

To run predictions on a custom drawing file (e.g., a handwritten number drawn and saved as a PNG/JPG):
```bash
python3 src/inference.py --image path/to/digit_image.png
```
*Note: The CLI preprocessing automatically detects if the background is light (black ink on white paper) and inverts the intensities to white-on-black strokes to ensure high-accuracy predictions matching the training set.*

### 4. Launching the Interactive Web App
Launch the grading assistant server locally:
```bash
python3 src/server.py
```
Open your web browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

* **Web UI Preprocessing Highlight:** The HTML5 Canvas frontend automatically performs **standard MNIST-style normalization** before sending pixels to the server. It finds the bounding box of your drawing, rescales it to fit a $20 \times 20$ grid while maintaining aspect ratio, and centers it inside a $28 \times 28$ matrix. A **"Feature Preview"** box displays the exact pixel values the model receives, yielding near-perfect, rotation-tolerant predictions!

---

## 🧠 Core Preprocessing & Math Concepts

### Pixel Normalization
Raw pixel intensities are stored as 8-bit integers ranging from `0` (white background) to `255` (black ink). High input values cause distance calculation saturation in SVM kernels. We scale the features down to `[0.0, 1.0]` using:
$$X_{\text{norm}} = \frac{X}{255.0}$$

### The SVM RBF Kernel
The Radial Basis Function (RBF) kernel is defined as:
$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
Where $\gamma$ controls the radius of influence of individual support vectors. By projecting the flattened 784-dimensional pixel arrays into an infinite-dimensional Hilbert space, the SVM finds an optimal separating hyperplane for overlapping handwritten strokes that are non-linear in their raw coordinates.

### Error Patterns
Our error analysis (visualized in `images/confusion_matrix.png`) identifies that the model is most prone to misclassifications between:
* **Digit 4 vs. 9:** Slanted, closed-loop 4s resemble the vertical tail and upper loop of a 9.
* **Digit 3 vs. 8:** Connecting the left open curves of a 3 with a light stroke transforms its Euclidean signature into an 8.
These findings illustrate that raw pixel distance classification is sensitive to translation, rotation, and line slants, pointing to **Histogram of Oriented Gradients (HOG)** or **Convolutional Neural Networks (CNNs)** as next-generation upgrades.