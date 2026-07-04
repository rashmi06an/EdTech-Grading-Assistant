import os
import time
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Trains a baseline Linear SVM classifier.
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training features and labels.
    X_test, y_test : numpy.ndarray
        Testing features and labels.
        
    Returns:
    --------
    model : SVC
        The fitted Linear SVM model.
    y_pred : numpy.ndarray
        Predictions on the test set.
    elapsed : float
        Time taken to train in seconds.
    """
    logger.info("Training baseline Linear SVM (kernel='linear', C=1.0)...")
    start_time = time.time()
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"Baseline Linear SVM trained in {elapsed:.2f} seconds.")
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Baseline Linear SVM Accuracy: {acc:.4f}")
    return model, y_pred, elapsed

def train_rbf_svm(X_train, y_train, C=1.0, gamma='scale'):
    """
    Trains an RBF-kernel SVM classifier.
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training features and labels.
    C : float
        Regularization parameter.
    gamma : str or float
        Kernel coefficient.
        
    Returns:
    --------
    model : SVC
        The fitted RBF SVM model.
    elapsed : float
        Time taken to train in seconds.
    """
    logger.info(f"Training RBF SVM (C={C}, gamma={gamma})...")
    start_time = time.time()
    # probability=True allows predicting class probabilities for the demo charts
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"RBF SVM trained in {elapsed:.2f} seconds.")
    return model, elapsed

def run_grid_search(X_train, y_train, param_grid=None, cv=3, subset_size=5000):
    """
    Performs hyperparameter sweep on SVM RBF using GridSearchCV.
    Uses a stratified subset of training data to keep training time low if specified.
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training features and labels.
    param_grid : dict, optional
        Parameters to sweep.
    cv : int
        Number of cross-validation folds.
    subset_size : int, optional
        Subsample size for grid search to keep execution under one minute.
        
    Returns:
    --------
    grid_search : GridSearchCV
        The fitted GridSearchCV object.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 'scale']
        }
    
    # Subsample if necessary to prevent long hangs during training sweeps
    if subset_size and X_train.shape[0] > subset_size:
        logger.info(f"Subsampling {subset_size} stratified samples for fast Grid Search.")
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, 
            train_size=subset_size, 
            stratify=y_train, 
            random_state=42
        )
    else:
        X_sub, y_sub = X_train, y_train
        
    logger.info(f"Running GridSearchCV on {X_sub.shape[0]} samples with grid: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=SVC(kernel='rbf', random_state=42),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_sub, y_sub)
    elapsed = time.time() - start_time
    
    logger.info(f"Grid Search completed in {elapsed:.2f} seconds.")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search

def evaluate_classifier(model, X_test, y_test, class_names=None):
    """
    Computes classification metrics: Accuracy, Precision, Recall, F1-score (macro).
    
    Parameters:
    -----------
    model : fitted estimator
        The classifier to evaluate.
    X_test, y_test : numpy.ndarray
        Testing features and labels.
    class_names : list of str, optional
        Names of target classes.
        
    Returns:
    --------
    y_pred : numpy.ndarray
        Predictions.
    acc, precision, recall, f1 : float
        Scalar metrics.
    report : dict
        Full metrics report dictionary.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    logger.info(f"Evaluation - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    return y_pred, acc, precision, recall, f1, report

def plot_confusion_matrix(y_true, y_pred, output_path, class_names=None):
    """
    Generates a Seaborn confusion matrix heatmap and saves it.
    
    Parameters:
    -----------
    y_true, y_pred : numpy.ndarray
        Ground truth and predictions.
    output_path : str
        Filepath to save the plot.
    class_names : list of str, optional
        Names of target classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names if class_names else [str(i) for i in range(10)],
        yticklabels=class_names if class_names else [str(i) for i in range(10)]
    )
    plt.title('MNIST Digit Classification Confusion Matrix Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Digit Label', fontsize=12)
    plt.xlabel('Predicted Digit Label', fontsize=12)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix heatmap saved successfully to: {output_path}")

def save_model(model, filepath):
    """
    Saves the trained model to disk.
    
    Parameters:
    -----------
    model : estimator
        Model to save.
    filepath : str
        Output path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a model from disk.
    
    Parameters:
    -----------
    filepath : str
        Model file path.
        
    Returns:
    --------
    model : estimator
        Loaded model object.
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model
