import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook():
    notebook_path = "notebooks/02_model_training_and_evaluation.ipynb"
    print(f"Loading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # Configure execution preprocessor
    # timeout=600 seconds (10 minutes) to allow full training to run comfortably
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    print("Executing notebook cells... This will run all training blocks (baseline, grid search, SVM, Random Forest).")
    try:
        # Run execution relative to notebooks directory so paths like ../src and ../models work
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        print("Notebook execution completed successfully!")
        
        # Save executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Executed notebook saved in-place to: {notebook_path}")
        
    except Exception as e:
        print(f"Error during notebook execution: {e}")
        raise e

if __name__ == "__main__":
    run_notebook()
