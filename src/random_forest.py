import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Trains a RandomForestClassifier on image features.
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training features and labels.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of the trees.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    model : RandomForestClassifier
        The fitted Random Forest model.
    elapsed : float
        Time taken to train in seconds.
    """
    logger.info(f"Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"Random Forest trained in {elapsed:.2f} seconds.")
    return model, elapsed

def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluates the Random Forest model and measures prediction speed.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained Random Forest model.
    X_test, y_test : numpy.ndarray
        Testing features and labels.
        
    Returns:
    --------
    acc, precision, recall, f1 : float
        Evaluation metrics.
    pred_time : float
        Time taken to predict on test set in seconds.
    """
    logger.info("Evaluating RandomForestClassifier...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    logger.info(f"Random Forest - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Random Forest prediction speed: {pred_time:.4f} seconds for {len(X_test)} samples.")
    
    return acc, precision, recall, f1, pred_time
