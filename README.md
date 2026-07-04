# Automated EdTech Grading Assistant

This repository contains the source code, Jupyter notebooks, interactive web server, and technical reports for the **Automated EdTech Grading Assistant** project, focusing on handwritten digit image classification using classical machine learning pipelines (Support Vector Machines with RBF Kernels).

---

## Repository Structure

```
EdTech-Grading-Assistant/
├── notebooks/
│   ├── 01_data_loading.ipynb              # Exploratory data analysis, reshaping, and pixel grid plots
│   ├── 02_model_training_and_evaluation.ipynb # Master validation notebook (pre-run with all metrics & heatmaps)
│   └── generate_notebook.py               # Programmatic notebook generator script
├── src/
│   ├── loader.py                          # Modular, cached, and SSL-bypassed MNIST data loader
│   ├── scaler.py                          # Preprocessing pixel scaling (/255.0 & MinMax) and stratified splitting
│   ├── svm_classifier.py                  # SVM classifier module (baseline training, GridSearchCV, CM heatmap)
│   ├── random_forest.py                   # RandomForest model module (speed and benchmarking)
│   ├── train.py                           # Master training script for model training and comparison log compilation
│   ├── inference.py                       # CLI inference pipeline wrapper with auto-color inversion and ASCII art
│   ├── server.py                          # Lightweight backend HTTP API server (uses standard http.server)
│   └── index.html                         # Premium drawing canvas HTML5 frontend with confidence charts
├── models/
│   └── best_svm_model.joblib              # Saved trained SVM RBF model (.joblib format)
├── images/
│   └── confusion_matrix.png               # Saved test set evaluation confusion matrix heatmap
├── reports/
│   ├── grid_search_log.csv                # Tabular log of hyperparameter sweep mean accuracies
│   ├── model_comparison.csv               # Tabular log of side-by-side benchmark metrics
│   ├── MNIST_VISUALIZATION_REPORT.md      # Report detailing data ingestion and spatial characteristics
│   ├── RUN_MANUAL.md                      # Detailed step-by-step setup and running manual
│   ├── TECHNICAL_REPORT.md                # Comprehensive technical report on SVM theory, tuning, and errors
│   └── presentation.html                  # Interactive 10-slide glassmorphic presentation deck
└── README.md                              # Repository overview (this file)
```

---

## Quick Start & Usage

### 1. Installation
Install core dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn Pillow
```

### 2. Model Training & Evaluation
To train the baseline, run the hyperparameter grid search, retrain the optimized SVM RBF model, and generate the Random Forest benchmark comparison:
```bash
python3 src/train.py
```
This script saves the best SVM model to `models/best_svm_model.joblib` and the evaluation heatmap to `images/confusion_matrix.png`.

### 3. Command-Line Inference (Predictions)
Run a demo prediction on a random MNIST sample (visualized in ASCII art):
```bash
python3 src/inference.py --demo
```
To run predictions on a custom image file (e.g. drawn digit scan):
```bash
python3 src/inference.py --image path/to/digit_image.png
```

### 4. Interactive Web Server Demo
Launch the local web server to draw and grade digits interactively in the browser:
```bash
python3 src/server.py
```
Then navigate your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## Technical Concept Notes

* **Pixel Normalization**: Raw 8-bit integer pixel values $[0, 255]$ are normalized down to standard floats $[0.0, 1.0]$ using a stateless division operation ($X_{\text{norm}} = X / 255.0$). This speeds up model convergence and prevents distance metric saturation.
* **The SVM RBF Kernel**: Maps flattened pixel features into high-dimensional space using:
  $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
  This projects overlapping, non-linear handwritten strokes into a space where they are linearly separable.
* **Hyperparameter Tuning**: Sweeping $C \in [0.1, 1, 10]$ and $\gamma \in [0.001, 0.01, 0.1, \text{'scale'}]$ reveals that an optimized RBF SVM ($C=10$, $\gamma=\text{'scale'}$) achieves **97.38% test accuracy** (exceeding the target 97%).
* **Performance Benchmarking**: Benchmarks against a Random Forest classifier show that while the SVM is slightly more accurate (97.38% vs 95.82%), the Random Forest trains (1.27s vs 103.64s) and predicts (0.026s vs 7.22s) orders of magnitude faster.