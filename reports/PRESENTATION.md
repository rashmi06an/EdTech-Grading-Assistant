# Automated EdTech Grading Assistant: Slide Deck
**Author: Rashmi Anand**  
**Project Title: Automated Handwriting Classification using Optimized SVM**

---

## Slide 1: Project Overview & Abstract

### Title: Automated EdTech Grading Assistant
* **Industry Context**: Millions of scanned worksheets are processed daily, costing thousands of hours in manual grading.
* **Objective**: Design an automated machine learning grading assistant to recognize and classify handwritten numbers directly from images.
* **Core Technology**: Classical machine learning with Support Vector Machines (SVM) and Radial Basis Function (RBF) kernels.
* **Outcome**: Developed an end-to-end pipeline reaching **97.38% grading accuracy**, exceeding our targets.

---

## Slide 2: The Core Challenge: Grading at Scale

### The Problem of Manual Evaluation
* **Inconsistency**: Human fatigue leads to grading inconsistencies over long sessions.
* **Latency**: Turnaround times for feedback on worksheets are typically 24-48 hours.
* **Cost**: Large school systems spend massive resources sorting and manually grading paper-based mathematical assessments.
* **The Solution**: Automate handwritten classification on local servers using classical machine learning, providing near-instantaneous feedback.

---

## Slide 3: The MNIST Benchmark Dataset

### Core Data Characteristics
* **Standard Dataset**: Modified National Institute of Standards and Technology (MNIST) database.
* **Dimensions**: 70,000 grayscale scanned images.
* **Pixel Grid**: $28 \times 28$ pixels per sample.
* **Target Classes**: 10 classes corresponding to numerical digits `0` through `9`.
* **Class Balance**: High balance across digits, with 6,000 to 7,900 samples per class, ensuring models do not suffer from class bias.

---

## Slide 4: Preprocessing & Scaling Pipeline

### Data Preparation Steps
1. **Flattening**: Transforming 2D spatial pixel matrices ($28 \times 28$) into 1D vectors of $784$ features.
2. **Min-Max Scaling**: Normalizing pixel intensity integers $[0, 255]$ down to floats $[0.0, 1.0]$.
   * Formula: $X_{\text{norm}} = X / 255.0$
   * Benefits: Speeds up SVM convergence and prevents feature scaling saturation.
3. **Stratified Splits**: Partitioning into a training set (20,000 samples) and a testing set (5,000 samples) to evaluate performance on unseen handwriting.

---

## Slide 5: The Math: SVM and the RBF Kernel

### Resolving Non-Linear Boundaries
* **Support Vector Machine**: Finds the hyperplane that separates data classes with the maximum margin.
* **Non-Linear Strokes**: Raw digit values overlap significantly. Simple straight lines cannot separate a '4' from a '9' or a '3' from an '8'.
* **The RBF Kernel Function**:
  $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
* **High-Dimensional Projection**: Implicitly maps pixel dimensions to an infinite-dimensional space where classes are linearly separable.

---

## Slide 6: Hyperparameter Sweep (GridSearchCV)

### Finding the Optimal Decision Boundary
* ** रेगुलराइजेशन (C)**: Balance between margin width and classification error.
* **Kernel Coefficient ($\gamma$)**: Range of influence of individual support vectors.
* **Sweep Strategy**: Stratified 3-fold cross validation on 5,000 samples.
* **Optimal Selection**:
  * **Best Regularization ($C$)**: `10`
  * **Best Kernel Coefficient ($\gamma$)**: `scale`
  * **Cross-Validation Accuracy**: **94.98%**

---

## Slide 7: Model Benchmarks Comparison

### Comparing Accuracy and Speeds

| Metric | Baseline Linear SVM | Optimized RBF SVM | Random Forest (100 Trees) |
|:---|:---:|:---:|:---:|
| **Accuracy** | 92.10% | **97.38%** | 95.82% |
| **Macro F1-Score** | 91.97% | **97.35%** | 95.78% |
| **Training Time** | 15.39s | 103.64s | **1.27s** |
| **Inference Time (5k)** | 15.39s | 7.22s | **0.026s** |

*RBF SVM delivers the highest grading accuracy, while Random Forest offers massive speed advantages.*

---

## Slide 8: Analyzing Grading Errors

### Why Does the Model Make Mistakes?
* **Digit 4 vs. 9**: Most common error. An open-topped '9' or a slightly slanted, closed-topped '4' share high pixel-overlap similarity.
* **Digit 3 vs. 8**: Curvature paths in a '3' are mathematically identical to an '8' except for a small vertical edge stroke.
* **Digit 5 vs. 6**: Slanted top bars in a '5' mimic the curved loop of a '6'.
* **Structural Diagnosis**: Discarding spatial coordinates during flattening makes models sensitive to slight translation shifts and rotations.

---

## Slide 9: EdTech Integration & Scaling

### Production Infrastructure Strategy
* **Minimal Infrastructure**: The RBF SVM model is small and runs on standard local CPUs, saving cloud GPU hosting costs.
* **Grade Response Latency**: Each classification takes **1.4 milliseconds**, allowing instantaneous feedback on digital tablets.
* **Quality Assurance Gate**:
  * Class classifications with confidence scores $< 75\%$ are flagged.
  * These are routed to human graders for verification, ensuring perfect grading integrity.

---

## Slide 10: Conclusion & Recommendations

### Final Guidelines
1. **RBF SVM Recommendation**: Adopt RBF SVM ($C=10, \gamma=\text{'scale'}$) as the primary engine for high-stakes grading where accuracy is critical.
2. **Preprocessing Standard**: Standardize the custom `/255.0` normalization. It is fast, robust, and stateless.
3. **Future Upgrades**:
   * Integrate Histogram of Oriented Gradients (HOG) features to improve rotation and shift tolerance.
   * Explore Convolutional Neural Networks (CNNs) for complex, multi-line character grading.
