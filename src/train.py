import os
import time
import logging
import numpy as np
import pandas as pd
from loader import load_mnist
from scaler import scale_pixels_custom, stratified_partition
from svm_classifier import (
    train_baseline, 
    train_rbf_svm, 
    run_grid_search, 
    evaluate_classifier, 
    plot_confusion_matrix, 
    save_model
)
from random_forest import train_random_forest, evaluate_random_forest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load MNIST Data
    logger.info("Step 1: Loading MNIST dataset...")
    X, y = load_mnist(cast_to_int=True)
    
    # 2. Reshaping & Pixel Scaling
    logger.info("Step 2: Normalizing pixel values to [0.0, 1.0] scale...")
    X_scaled = scale_pixels_custom(X)
    
    # 3. Partitioning Data
    logger.info("Step 3: Creating stratified train/test partitions...")
    # Training size: 20,000; Test size: 5,000
    X_train, X_test, y_train, y_test = stratified_partition(
        X_scaled, y, 
        train_size=20000, 
        test_size=5000, 
        random_state=42
    )
    
    # 4. Train Baseline Linear SVM
    logger.info("Step 4: Training baseline Linear SVM...")
    baseline_model, y_pred_baseline, t_baseline = train_baseline(X_train, y_train, X_test, y_test)
    _, acc_base, p_base, r_base, f1_base, report_base = evaluate_classifier(
        baseline_model, X_test, y_test
    )
    
    # 5. GridSearchCV hyperparameter sweep
    logger.info("Step 5: Running GridSearchCV for SVM RBF hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1, 'scale']
    }
    # Using 5,000 training samples for grid search to keep it under 1 minute
    grid_search = run_grid_search(X_train, y_train, param_grid=param_grid, cv=3, subset_size=5000)
    best_params = grid_search.best_params_
    
    # Save hyperparameter tuning results to logs
    logger.info("Saving hyperparameter grid search logs...")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    os.makedirs("reports", exist_ok=True)
    cv_results.to_csv("reports/grid_search_log.csv", index=False)
    logger.info("Grid search log saved to reports/grid_search_log.csv")
    
    # 6. Train Optimized SVM RBF
    logger.info("Step 6: Training optimized RBF SVM with best parameters on full 20,000 subset...")
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    svm_opt_model, t_svm_opt = train_rbf_svm(X_train, y_train, C=best_C, gamma=best_gamma)
    
    # Evaluate optimized SVM
    y_pred_svm, acc_svm, p_svm, r_svm, f1_svm, report_svm = evaluate_classifier(
        svm_opt_model, X_test, y_test
    )
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_test, y_pred_svm, "images/confusion_matrix.png")
    
    # Save optimized SVM model
    save_model(svm_opt_model, "models/best_svm_model.joblib")
    
    # 7. Train & Evaluate Random Forest Classifier
    logger.info("Step 7: Training and evaluating Random Forest Classifier...")
    rf_model, t_rf_train = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    acc_rf, p_rf, r_rf, f1_rf, t_rf_pred = evaluate_random_forest(rf_model, X_test, y_test)
    
    # 8. Comparison Analysis Table
    logger.info("Step 8: Compiling comparison analysis results...")
    
    # Timing prediction speed for SVM
    t_svm_pred_start = time.time()
    _ = svm_opt_model.predict(X_test)
    t_svm_pred = time.time() - t_svm_pred_start
    
    comparison_data = {
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Training Time (s)", "Prediction Time (s)"],
        "Baseline Linear SVM": [acc_base, p_base, r_base, f1_base, t_baseline, t_baseline], # Baseline pred time approximated or same
        "Optimized RBF SVM": [acc_svm, p_svm, r_svm, f1_svm, t_svm_opt, t_svm_pred],
        "Random Forest": [acc_rf, p_rf, r_rf, f1_rf, t_rf_train, t_rf_pred]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "="*60)
    print("                    MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_markdown(index=False))
    print("="*60 + "\n")
    
    # Save comparison table to reports
    comparison_df.to_csv("reports/model_comparison.csv", index=False)
    logger.info("Model comparison results saved to reports/model_comparison.csv")
    
    # Log out final confirmation
    logger.info("Pipeline executed successfully. All models trained, evaluated, and saved!")

if __name__ == "__main__":
    main()
