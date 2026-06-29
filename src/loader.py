import ssl
import logging
import numpy as np
from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_mnist(version=1, as_frame=False, cast_to_int=True):
    """
    Fetches the MNIST handwritten digits dataset from OpenML.
    
    Parameters:
    -----------
    version : int, optional (default=1)
        The version of the MNIST dataset to load.
    as_frame : bool, optional (default=False)
        If True, returns the data as a pandas DataFrame.
        If False (default), returns the data as NumPy arrays.
    cast_to_int : bool, optional (default=True)
        If True, casts the target labels from strings/categories to 8-bit integers.
        
    Returns:
    --------
    X : numpy.ndarray or pandas.DataFrame
        The feature matrix of shape (70000, 784).
    y : numpy.ndarray or pandas.Series
        The target labels of shape (70000,).
    """
    # Bypass SSL verification check for OpenML fetch (common issue on macOS)
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info("Bypassed SSL context verification for OpenML dataset retrieval.")
    except Exception as e:
        logger.warning(f"Could not bypass SSL context: {e}")

    logger.info("Fetching MNIST dataset (mnist_784) from OpenML. This might take a moment...")
    try:
        mnist = fetch_openml('mnist_784', version=version, as_frame=as_frame, parser='liac-arff')
        X, y = mnist["data"], mnist["target"]
        
        if cast_to_int:
            y = y.astype(np.int8)
            logger.info("Target labels cast to 8-bit integers.")
            
        logger.info(f"Successfully loaded MNIST. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to fetch MNIST dataset from OpenML: {e}")
        raise e

if __name__ == "__main__":
    # Test execution when run as a script
    X, y = load_mnist()
    print(f"Features: min={X.min()}, max={X.max()}, shape={X.shape}, dtype={X.dtype}")
    print(f"Labels: unique={np.unique(y)}, shape={y.shape}, dtype={y.dtype}")
