# Technical Report: Automated EdTech Grading Assistant
**Author: Rashmi Anand**  
**Date: June 29, 2026**

---

## 1. Executive Summary
Grading large volumes of scanned, handwritten numerical worksheets is a standard challenge in modern educational systems. This project implements a classical machine learning pipeline using Support Vector Machines (SVM) with Radial Basis Function (RBF) kernels to automate the classification of handwritten digits (0–9) using the MNIST database. 

By training on a stratified subset of 20,000 images and testing on 5,000 images, our optimized RBF SVM classifier achieved an **accuracy of 97.38%**, exceeding the target project benchmark of 97%. This report provides a detailed breakdown of the mathematical framework, data engineering pipeline, tuning sweeps, and comparative analyses against baseline linear models and Random Forest ensembles.

---

## 2. Theoretical Framework: Support Vector Machines & The RBF Kernel

### 2.1 The Support Vector Machine Paradigm
Support Vector Machines are discriminative classifiers that seek to identify the optimal hyperplane that separates classes with the maximum margin. For linearly separable data in a feature space, the decision boundary is given by:
$$w^T x + b = 0$$
where $w$ represents the weight vector perpendicular to the hyperplane and $b$ is the bias. The optimization problem maximizes the margin ($2/\|w\|$) subject to class separation constraints:
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1 \quad \forall i$$

### 2.2 The "Kernel Trick" & High-Dimensional Mapping
In complex datasets like MNIST, digit pixel stroke boundaries are highly non-linear and overlapping in the raw 784-dimensional space. To resolve this, Support Vector Machines utilize a kernel function $K(x_i, x_j)$ to implicitly map the input features into a much higher-dimensional Hilbert space where the data becomes linearly separable. 

The **Radial Basis Function (RBF) Kernel** (also known as the Gaussian kernel) is defined mathematically as:
$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
where $x_i$ and $x_j$ are two feature vectors, and $\gamma$ is a kernel coefficient parameter (where $\gamma > 0$).

### 2.3 The RBF Gamma ($\gamma$) Parameter
The $\gamma$ parameter controls the radius of influence of individual support vectors:
- **High $\gamma$**: The kernel function decays rapidly as distance increases. Individual support vectors have localized influence, leading to a complex, tight decision boundary. This can result in overfitting.
- **Low $\gamma$**: The kernel decays slowly. Support vectors have a wide radius of influence, creating a smoother, more generalized decision boundary. If too low, it can lead to underfitting.
- **`scale` Option**: Scikit-Learn sets $\gamma = 1 / (n\_features \times \text{variance}(X))$, which serves as an excellent standardized starting point.

---

## 3. Preprocessing and Scaling Pipeline

### 3.1 Feature Matrix Flattening
Each scanned digit is originally structured as a 2D grid of size $28 \times 28$ pixels. To input this into classical classifiers, the grid is flattened into a 1D vector of length 784 in row-major (C-style) ordering. The flattening process discards 2D coordinates but retains spatial relationships implicitly in the sequence of pixel intensities.

### 3.2 Min-Max Normalization
Raw pixel values are represented as 8-bit integers ranging from `0` (white background) to `255` (black ink). High-valued raw pixel inputs cause distance-based models (such as SVMs) to saturate and slow down convergence.
We normalize the pixel intensities from $[0, 255]$ down to $[0.0, 1.0]$ using:
$$X_{\text{norm}} = \frac{X}{255.0}$$
This linear min-max scaling provides several key benefits:
1. **Speed of Convergence**: Standardized features ensure that gradients propagate uniformly during optimization, preventing numerical instability.
2. **Support Vector Stability**: It keeps distance metrics $\|x_i - x_j\|^2$ bounded, preventing high-valued dimensions from dominating decision boundary computations.
3. **Stateless Production Integration**: Simple division by `255.0` is a stateless operation, removing the need to export and load fitted scaling objects, which eliminates a common source of bugs in production deployment.

### 3.3 Stratified Partitioning
To ensure our test metrics represent the general population, data splitting must maintain the target labels' class proportions. We partitioned our 70,000 samples into:
- **Training Subset**: 20,000 stratified samples (keeps training time under 2 minutes).
- **Testing Subset**: 5,000 stratified samples (sufficient to obtain tight confidence intervals).

---

## 4. Hyperparameter Optimization Tune Logs
We performed a 3-fold cross-validated grid search (`GridSearchCV`) using a stratified subset of 5,000 samples to tune the regularization parameter $C$ and the kernel coefficient $\gamma$.

### 4.1 Regularization Parameter ($C$)
The parameter $C$ controls the trade-off between maximizing the decision boundary margin and minimizing training errors:
- **Low $C$**: Allows a larger margin with some misclassified training points, encouraging generalization (high bias, low variance).
- **High $C$**: Penalizes training errors heavily, forcing the optimizer to fit training points precisely (low bias, high variance).

### 4.2 Grid Search Results Sweep Log
The mean validation accuracies across our 3-fold cross validation grid search are tabulated below:

| Regularization ($C$) | Kernel Coefficient ($\gamma$) | Mean CV Accuracy |
|:---------------------|:------------------------------|-----------------:|
| 0.1                  | 0.001                         |           0.6482 |
| 0.1                  | 0.01                          |           0.8214 |
| 0.1                  | 0.1                           |           0.2238 |
| 0.1                  | scale                         |           0.8988 |
| 1.0                  | 0.001                         |           0.9090 |
| 1.0                  | 0.01                          |           0.9416 |
| 1.0                  | 0.1                           |           0.5842 |
| 1.0                  | scale                         |           0.9452 |
| **10.0**             | 0.001                         |           0.9272 |
| **10.0**             | 0.01                          |           0.9492 |
| **10.0**             | 0.1                           |           0.6062 |
| **10.0**             | **scale**                     |     **0.9498**   |

### Key Takeaways from Hyperparameter Sweep:
1. **Best Hyperparameters**: $C = 10$, $\gamma = \text{'scale'}$ yielded the highest mean cross-validation accuracy of **94.98%**.
2. **Sensitivity to Gamma**: When $\gamma = 0.1$, accuracy drops severely (down to 22.38% for $C=0.1$ and 60.62% for $C=10$). This is because a high gamma value creates overly localized decision regions (hyper-ellipsoids around individual support vectors), causing severe overfitting where almost all test points fall outside the decision boundaries.
3. **Role of C**: Increasing $C$ from $0.1$ to $10.0$ consistently improved classification boundaries, demonstrating that a slightly narrower margin with a higher penalty on training errors is necessary to separate the overlaps in digit strokes.

---

## 5. Model Evaluation and Performance Benchmarking

After hyperparameter selection, we retrained the RBF SVM on the full 20,000 sample training set and benchmarked its performance against the baseline Linear SVM and a RandomForest Classifier.

### 5.1 Performance Benchmarking Table

| Metric | Baseline Linear SVM | Optimized RBF SVM ($C=10, \gamma=\text{'scale'}$) | Random Forest (100 Trees) |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 92.10% | **97.38%** | 95.82% |
| **Macro Precision** | 92.03% | **97.36%** | 95.78% |
| **Macro Recall** | 91.97% | **97.36%** | 95.79% |
| **Macro F1-Score** | 91.97% | **97.35%** | 95.78% |
| **Training Time (s)** | 15.39s | 103.64s | **1.27s** |
| **Inference Time (5k samples)** | 15.39s | 7.22s | **0.026s** |

### 5.2 Critical Analysis of Benchmarking Trade-offs
1. **Accuracy Advantage**: The Optimized RBF SVM achieves the highest accuracy (**97.38%**), outperforming the baseline Linear SVM by over 5.2% and Random Forest by 1.5%. This validates that handwritten digit strokes are highly non-linear and benefit significantly from RBF high-dimensional projection.
2. **The Speed Trade-Off (Training)**: The Random Forest ensemble trained in just **1.27 seconds**, which is nearly 80x faster than the RBF SVM (103.64s). The RBF SVM's high training cost is due to calculating the kernel matrix pairwise across all training samples, which scales quadratically with dataset size ($O(N^2)$).
3. **The Inference Bottleneck**: While Random Forest classifies 5,000 images in **0.026 seconds** (~5 microseconds per image), the RBF SVM takes **7.22 seconds** (~1.4 milliseconds per image). For production applications grading millions of documents daily, Random Forest is far more computationally efficient, though slightly less accurate. RBF SVM remains preferred when grading accuracy is paramount.

---

## 6. Error Analysis and Confusion Matrix Heatmap

The test set predictions were analyzed using a confusion matrix heatmap to identify the classes most frequently misclassified by our optimized SVM RBF model.

### 6.1 Common Misclassification Patterns
Our confusion matrix analysis shows that the most frequent errors occur between specific digit pairs:
- **Digit 4 vs. Digit 9**: 4s are frequently misclassified as 9s and vice versa. This occurs because the top loop of a 9 can look identical to the open top of a 4 if written with a slight slant or a closed top stroke.
- **Digit 3 vs. Digit 8**: 3s are occasionally misclassified as 8s. This is structurally intuitive, as adding a small vertical stroke connecting the left side of a 3 transforms it into an 8.
- **Digit 5 vs. Digit 6**: A 5 with a curved upper stroke and a closed-loop lower loop resembles a 6.
- **Digit 7 vs. Digit 9**: A 7 with a long vertical stroke and a curved horizontal top line matches the layout of a 9 without a fully closed top loop.

### 6.2 Spatial Feature Explanations
These misclassifications highlight the limitations of raw pixel features. Because flattening discards explicit geometric concepts (such as lines, curves, loops, and intersections), the model relies purely on pixel-to-pixel overlap distances. When strokes shift by a few pixels due to writing style, slant, or alignment, the Euclidean distance increases, resulting in misclassifications. 

*Recommendation*: Integrating spatial descriptors (such as HOG - Histogram of Oriented Gradients) or using Convolutional Neural Networks (CNNs) would mitigate these coordinate-shift issues by extracting translation-invariant features.

---

## 7. Educational Grading Impact & Scalability
Implementing this classification pipeline within an EdTech platform delivers substantial operational improvements:
- **Latency Reduction**: Manual grading of a 30-question math worksheet typically takes 2–3 minutes per student. Our RBF SVM server classifies each digit in **1.4 milliseconds**, enabling near-instant grading of scanned worksheets.
- **Scalability**: Classical models can run on standard CPUs (such as a MacBook Pro M2), eliminating the need for expensive GPU infrastructure required by modern Deep Learning models.
- **Feedback Quality**: By mapping class probabilities (probabilities of digits 0-9), the grading system can flag low-confidence classifications (e.g., confidence < 75%) for manual review, maintaining high accuracy while automating 95%+ of the workload.
