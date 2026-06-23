# Project Proposal: Automated EdTech Grading Assistant Using Handwritten Digit Classification

## Student Name

Rashmi Anand

## Project Title

Automated EdTech Grading Assistant (Handwritten Image Classification using Support Vector Machines)

## Abstract

Educational institutions process large volumes of handwritten assignments and worksheets that require manual evaluation. This project aims to develop an automated handwritten digit recognition system capable of accurately classifying numerical digits from scanned images. Using the MNIST Handwritten Digits dataset, the system will preprocess image data, transform pixel matrices into numerical feature vectors, and train a multi-class Support Vector Machine (SVM) classifier with an RBF kernel. The project will also perform model evaluation, hyperparameter optimization, error analysis, and comparative benchmarking against alternative machine learning models.

## Problem Statement

Manual grading of handwritten worksheets is time-consuming and difficult to scale. The objective of this project is to design an image classification pipeline that automatically recognizes handwritten numerical digits and predicts their correct labels with high accuracy.

## Objectives

* Load and preprocess handwritten digit image data.
* Convert image matrices into flattened feature vectors.
* Normalize pixel intensity values using Min-Max scaling.
* Train a baseline classification model.
* Develop an optimized SVM classifier using an RBF kernel.
* Analyze classification errors through confusion matrices.
* Tune model hyperparameters using GridSearchCV.
* Compare SVM performance with Random Forest classifiers.
* Develop an inference pipeline for unseen handwritten digits.

## Dataset

Dataset Name: MNIST Handwritten Digits Database

Characteristics:

* 70,000 grayscale handwritten digit images
* Image resolution: 28 × 28 pixels
* 10 output classes (digits 0–9)
* 784 numerical features after flattening

Source:
Scikit-Learn OpenML API

## Methodology

1. Data Collection and Exploration
2. Image Visualization and Feature Analysis
3. Pixel Matrix Flattening
4. Feature Scaling and Normalization
5. Train-Test Data Partitioning
6. Baseline Model Development
7. SVM Model Training with RBF Kernel
8. Hyperparameter Optimization
9. Error Analysis using Confusion Matrices
10. Comparative Evaluation with Random Forest
11. Deployment of Interactive Inference Pipeline

## Technology Stack

Language:

* Python 3.10+

Libraries:

* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

Development Environment:

* Jupyter Notebook
* VS Code

## Expected Outcomes

* Classification accuracy exceeding 97%.
* Optimized SVM model with tuned hyperparameters.
* Comprehensive confusion matrix analysis.
* Interactive handwritten digit prediction system.
* Complete end-to-end machine learning workflow.

## Deliverables

* Source Code Repository
* Data Preprocessing Pipeline
* Trained SVM Model
* Evaluation Reports
* Confusion Matrix Visualizations
* Technical Documentation
* Presentation Deck
* Demonstration Video

## Timeline

Week 1–2: Data Exploration and Preprocessing

Week 3–4: Baseline Model and SVM Development

Week 5–6: Evaluation and Hyperparameter Tuning

Week 7–8: Model Comparison, Deployment, Documentation, and Final Presentation

## Success Criteria

* Accuracy ≥ 97%
* Strong Precision, Recall, and F1 Score
* Successful prediction on unseen handwritten digit images
* Reproducible and modular machine learning pipeline
