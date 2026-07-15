import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def flatten_images(X):
    """
    Flattens a batch of 2D image arrays into 1D feature vectors.
    Converts images of shape (n, 28, 28) into (n, 784) for model input.
    If already flat (n, 784), returns as-is.

    Parameters:
    -----------
    X : numpy.ndarray
        Image array of shape (n_samples, 28, 28) or already (n_samples, 784).

    Returns:
    --------
    X_flat : numpy.ndarray
        Flattened feature matrix of shape (n_samples, 784).
    """
    if X.ndim == 3:
        n_samples = X.shape[0]
        logger.info(f"Flattening {n_samples} images from shape {X.shape} to ({n_samples}, 784).")
        return X.reshape(n_samples, -1)
    logger.info("Data already flat, skipping reshape.")
    return X

def scale_pixels_custom(X):
    """
    Normalizes pixel intensity values from [0, 255] down to [0.0, 1.0] using fast custom division.
    This is highly efficient for standard 8-bit image data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix containing raw pixel values (0 to 255).
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        The normalized feature matrix as 32-bit floats.
    """
    logger.info("Normalizing features via custom division (X / 255.0).")
    return X.astype(np.float32) / 255.0

def scale_pixels_minmax(X, scaler=None):
    """
    Normalizes pixel values using scikit-learn's MinMaxScaler.
    
    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix to scale.
    scaler : MinMaxScaler, optional (default=None)
        A pre-fitted MinMaxScaler. If None, a new scaler is fitted on X.
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        The scaled feature matrix.
    scaler : MinMaxScaler
        The fitted scaler instance (useful to transform test/unseen data).
    """
    if scaler is None:
        logger.info("Initializing and fitting new MinMaxScaler.")
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_scaled = scaler.fit_transform(X)
    else:
        logger.info("Transforming features using pre-fitted MinMaxScaler.")
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def stratified_partition(X, y, train_size=20000, test_size=5000, random_state=42):
    """
    Creates stratified training and testing partitions from the dataset.
    Ensures that the class distributions are maintained perfectly across splits.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target labels.
    train_size : int
        Number of training samples.
    test_size : int
        Number of testing samples.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        The partitioned features and labels.
    """
    total_requested = train_size + test_size
    n_samples = X.shape[0]
    
    logger.info(f"Partitioning data into stratified train ({train_size}) and test ({test_size}) subsets.")
    
    if total_requested > n_samples:
        raise ValueError(f"Requested total partition size ({total_requested}) exceeds dataset size ({n_samples}).")
        
    # Get the representative subset first using train_test_split (stratified)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, 
        train_size=total_requested, 
        stratify=y, 
        random_state=random_state
    )
    
    # Now split the subset into train and test sizes
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        train_size=train_size,
        stratify=y_sub,
        random_state=random_state
    )
    
    logger.info(f"Split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test script execution
    from loader import load_mnist
    X, y = load_mnist()
    
    # Preprocess
    X_scaled = scale_pixels_custom(X)
    X_train, X_test, y_train, y_test = stratified_partition(X_scaled, y, train_size=100, test_size=50)
    
    print(f"Sample train pixel range: min={X_train.min()}, max={X_train.max()}")
    assert np.all(X_train >= 0.0) and np.all(X_train <= 1.0)
    print("Self-test passed!")
