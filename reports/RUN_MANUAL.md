# Run Manual: Automated EdTech Grading Assistant

This document contains step-by-step instructions for installing dependencies, training the machine learning models, running predictions from the command line, and launching the interactive drawing canvas web interface.

---

## 1. Prerequisites and Installation

Ensure you have Python 3.10+ installed on your system. 

### Core Dependencies
Install the required scientific computing and visualization libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Additional Web Demo & CLI Dependencies
To load custom image files (PNG/JPG) using the command-line inference helper, you will need the Pillow library:
```bash
pip install Pillow
```

*Note: The web demo server is built using Python's standard `http.server` library and requires no third-party backend packages.*

---

## 2. Directory Structure

Ensure your workspace contains the following files in these exact paths:
```
EdTech-Grading-Assistant/
├── notebooks/
│   ├── 01_data_loading.ipynb              # EDA and Reshaping math
│   └── 02_model_training_and_evaluation.ipynb # Master training notebook
├── src/
│   ├── loader.py                          # MNIST dataset retriever
│   ├── scaler.py                          # Pixel normalizer and splitter
│   ├── svm_classifier.py                  # SVM classifier module
│   ├── random_forest.py                   # RandomForest model module
│   ├── train.py                           # Master training script
│   ├── inference.py                       # CLI inference wrapper
│   ├── server.py                          # Lightweight backend HTTP API server
│   └── index.html                         # Premium drawing canvas frontend UI
├── models/
│   └── best_svm_model.joblib              # Saved trained SVM RBF model (generated after training)
├── images/
│   └── confusion_matrix.png               # Saved evaluation plot (generated after training)
└── reports/
    ├── grid_search_log.csv                # CSV log of tuning sweeps (generated after training)
    ├── model_comparison.csv               # CSV log of benchmark results (generated after training)
    ├── RUN_MANUAL.md                      # Run instructions (this file)
    ├── TECHNICAL_REPORT.md                # Comprehensive project report
    └── PRESENTATION.md                    # 10-slide presentation deck
```

---

## 3. Training the Classifier Pipeline

To train the baseline, run the hyperparameter grid search, compile the comparison benchmark, and save the final optimized SVM model:

1. Open your terminal.
2. Navigate to the project root directory:
   ```bash
   cd /Users/rashmianand/Desktop/EdTech-Grading-Assistant
   ```
3. Run the master training script:
   ```bash
   python3 src/train.py
   ```

### What happens during training:
- **Data Load**: The script fetches the standard MNIST dataset (~70,000 rows).
- **Normalize**: Pixel intensities are normalized to the range `[0.0, 1.0]`.
- **Partition**: Splits the data into a stratified training subset (20,000 samples) and test subset (5,000 samples).
- **Baseline**: Trains a baseline Linear SVM classifier and logs its initial accuracy.
- **Tuning**: Runs a cross-validated grid search (`GridSearchCV` with 3 folds) on a stratified subset of 5,000 samples to select the best values for $C$ and $\gamma$. It saves the tuning sweep records to `reports/grid_search_log.csv`.
- **Optimization**: Retrains the optimized RBF SVM model using the best hyperparameters on the full 20,000 training set. The model is saved to `models/best_svm_model.joblib`.
- **Evaluation**: Renders the Seaborn confusion matrix heatmap and saves it to `images/confusion_matrix.png`.
- **Benchmarking**: Trains a `RandomForestClassifier` on the same training set, benchmarks its training and inference speed against the optimized SVM, and prints a comparative markdown table in the terminal. The summary is saved to `reports/model_comparison.csv`.

---

## 4. Running Command-Line Predictions (CLI Inference)

You can run predictions using the `src/inference.py` utility.

### Option A: Run a Demo Classification
Pick a random digit from the dataset, print its ASCII art visualization, and predict its value:
```bash
python3 src/inference.py --demo
```

### Option B: Classify a Custom Handwriting Image
To classify a custom digit drawn by hand and saved as an image:
```bash
python3 src/inference.py --image path/to/digit_image.png
```

*Note: The CLI preprocessing pipeline automatically detects white backgrounds (black ink on paper) and inverts the colors to white-on-black strokes to ensure high-accuracy predictions matching the MNIST training format.*

---

## 5. Launching the Interactive Web Interface

To host and run the interactive grading assistant locally:

1. Start the backend API server:
   ```bash
   python3 src/server.py
   ```
2. The terminal will show the server status:
   ```
   INFO - --- Automated EdTech Grading Server ---
   INFO - Local Server URL: http://localhost:8000
   INFO - Model successfully loaded from models/best_svm_model.joblib
   INFO - System status: READY for grading classification.
   ```
3. Open your web browser and navigate to:
   [http://localhost:8000](http://localhost:8000)
4. **Drawing and Grading**:
   - Draw a digit in the black canvas box using your mouse or trackpad.
   - The web app automatically predicts the digit and displays the confidence distribution bar charts in real-time as you release the brush.
   - Click **Grade** to explicitly trigger classification.
   - Click **Clear** to wipe the canvas and reset predictions.

---

## 6. Running Jupyter Notebooks

If you prefer to work interactively:
1. Start Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/01_data_loading.ipynb` to view the initial exploratory data analysis.
3. Open `notebooks/02_model_training_and_evaluation.ipynb` to run the model training, hyperparameter sweep, and evaluation blocks interactively within your notebook environment.
