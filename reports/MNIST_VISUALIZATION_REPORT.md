# Automated EdTech Grading Assistant: Ingestion & Spatial Visualization Report

This report provides a detailed explanation of the dataset ingestion, pixel matrix reshaping mechanics, and data visualization strategies implemented for the **Automated EdTech Grading Assistant** project.

---

## 1. Project Context & Objectives
Educational institutions and online learning platforms process millions of handwritten sheets daily. Automating the grading process is critical to scale and reduce manual labor. This project forms the foundational step of building a handwritten image classifier. 
Our primary objective in this phase is to:
* Safely ingest the standardized **MNIST Handwritten Digits Database** using Scikit-Learn.
* Map flat 1D pixel intensity arrays back into their natural 2D spatial matrices ($28 \times 28$).
* Visualize and analyze digit characters to understand spatial layout, pixel distribution, and dataset balance.

---

## 2. Ingesting the MNIST Dataset
The dataset is fetched from **OpenML** via Scikit-Learn's built-in fetch API:
```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
```

### Ingestion Details & Parameter Selection:
1. **`mnist_784`**: Refers to the MNIST database version containing 70,000 images of size $28 \times 28 = 784$ pixels.
2. **`version=1`**: Selects the standard default version of the MNIST dataset.
3. **`as_frame=False`**: Instructs Scikit-Learn to return the dataset as **NumPy arrays** instead of a Pandas DataFrame. Since we need to reshape raw feature rows into matrices, NumPy arrays are highly performant and avoid DataFrame indexing overhead.
4. **`parser='liac-arff'`**: Specifies the parser used to read the ARFF format file downloaded from OpenML.
5. **SSL Certificate Bypass**: On macOS, Python installations sometimes lack default root certificates, leading to connection errors. We handle this using the `ssl` module inside our setup cell:
   ```python
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context
   ```
6. **Caching**: Scikit-Learn caches the downloaded dataset locally in `~/scikit_learn_data/`. Any future executions will load the dataset instantaneously without making network calls.

---

## 3. Dataset Characteristics & Properties
Running exploratory analysis on the loaded arrays reveals the following metadata:

* **Feature Matrix ($X$) Shape**: `(70000, 784)`
  * 70,000 individual digit image samples.
  * Each sample has 784 features representing the flattened grayscale intensity values of a $28 \times 28$ grid.
* **Target Vector ($y$) Shape**: `(70000,)`
  * Contains the corresponding ground-truth digit label (0 to 9) for each image.
* **Data Types**:
  * $X$ is stored as floating-point or integer pixel intensities (`float64` / `int64`).
  * $y$ is initially loaded as categorical/object strings (e.g. `'0'`, `'1'`), which we cast to `int8` integers to save memory and facilitate statistical plots.
* **Intensity Range**: Raw values range from `0` (completely white/background) to `255` (completely black/foreground ink).

### Class Distribution
To ensure the dataset is balanced, we checked the class counts across all 10 digits (0-9). The dataset is well-balanced:
* **0**: 6,903 samples
* **1**: 7,877 samples
* **2**: 6,990 samples
* **3**: 7,141 samples
* **4**: 6,824 samples
* **5**: 6,313 samples
* **6**: 6,876 samples
* **7**: 7,293 samples
* **8**: 6,825 samples
* **9**: 6,958 samples

---

## 4. The Mechanics of Reshaping
Computers store images as contiguous arrays of numbers. For MNIST, each $28 \times 28$ image is flattened into a 1D vector of length 784.

### Row-Major Ordering (C-Style)
By default, NumPy handles reshaping in **row-major ordering** (also known as C-style ordering). This means the 1D array is divided sequentially into rows:
* The first 28 elements (indices 0 to 27) represent Row 0 of the image.
* The next 28 elements (indices 28 to 55) represent Row 1.
* In general, the pixel at row index $r$ and column index $c$ in the 2D matrix ($0 \le r, c < 28$) maps to index $i$ in the 1D flat vector via:
  $$i = r \times 28 + c$$

### Code Implementation
We implemented a utility function to automate this mapping:
```python
def reshape_digit(feature_row):
    return feature_row.reshape(28, 28)
```
Calling `X[0].reshape(28, 28)` maps the shape from `(784,)` to `(28, 28)`.

---

## 5. Visualizing Image Matrices
To understand the structure of the handwritten digits, we implemented multiple layers of visualizations using Matplotlib and Seaborn.

### 5.1 Single Digit Analysis with Pixel Intensities
For a granular look at how numerical intensities map to human-readable shapes, we plotted a sample digit using `matplotlib.pyplot.imshow` with annotations:
* **Colormap (`cmap='Purples'`)**: Maps intensity values from 0 (white) to 255 (deep purple).
* **Interpolation (`interpolation='nearest'`)**: Renders pixels as sharp squares without blending or smoothing, showing the exact raw pixel boundaries.
* **Aesthetic Grids**: Tick marks are set at every 2 units to read coordinate offsets easily.
* **In-cell Labels**: Each cell contains the text of its raw pixel intensity (0 to 255) for all cells where the intensity is greater than zero.

```python
# Conceptual rendering details
plt.imshow(digit_image, cmap='Purples', interpolation='nearest')
for i in range(28):
    for j in range(28):
        val = int(digit_image[i, j])
        if val > 0:
            plt.text(j, i, str(val), ...)
```
*Insight:* This annotation grid highlights the thick stroke center of the pen and the anti-aliased edge boundaries (values like 40 to 180) where the stroke fades.

### 5.2 Multi-Class Grids
We created two primary grids to explore handwriting styles:
1. **Random 25-Sample Grid**: Selected 25 random images using `numpy.random.choice` and plotted them with titles mapping their true labels, using `cmap='binary'`. This visualizes the variability of strokes (some write '7' with a crossbar, some write '1' as a simple vertical stroke).
2. **Class-Representative Matrix (10x5)**: Plots 5 distinct samples for every single digit class (0 through 9) in rows. This verifies that our dataset covers every category with representative variations. We styled this grid using the vibrant `plasma` colormap for high-contrast presentation.

---

## 6. Key Findings

Based on our exploratory data analysis and visualization, we observed several core features of the MNIST dataset:
* **Dataset Size**: The dataset contains 70,000 high-quality handwritten digit images.
* **Dimensionality**: Each image is represented by 784 individual pixel intensity values.
* **Spatial Relationship**: Applying 2D reshaping restores the original $28 \times 28$ spatial grid structure, turning a list of numbers back into a recognizable image.
* **Intensity Values**: Pixels are represented by integers ranging from 0 (completely white background) to 255 (completely black foreground ink).
* **Class Balance**: The dataset is relatively well-balanced across all 10 digit classes (0–9), with each class containing between 6,000 and 7,900 samples.
* **Writing Variance**: Handwriting styles vary significantly even within the same digit class (e.g., some write the number '7' with a crossbar, while others write it as a simple stroke; some draw '0' as a perfect circle, while others draw it as a narrow ellipse).

---

## 7. Evaluation Readiness

To prepare for a project evaluation, here are concise answers to fundamental dataset and processing questions:

* **Why does $X$ have the shape `(70000, 784)`?**
  $X$ contains 70,000 rows (one for each image) and 784 columns. Each column represents a single pixel from the flattened $28 \times 28$ grid ($28 \times 28 = 784$).
* **Why does $y$ have the shape `(70000,)`?**
  $y$ is a 1D label array matching the 70,000 image samples in $X$. Each element contains the ground-truth target label (the digit 0 through 9) corresponding to that image.
* **Why is reshaping required?**
  Computers load image pixels as a flat, 1D array of numbers for efficient storage and matrix computations. Reshaping is necessary to restore the 2D spatial context ($28 \times 28$), allowing coordinate-based plotting and spatial feature analysis.
* **What does a pixel value represent?**
  Each pixel value represents a grayscale intensity ranging from `0` to `255`. A value of `0` denotes the background (no ink/white), and `255` denotes the maximum foreground stroke intensity (heavy ink/black), with intermediate values representing the anti-aliased edge boundaries of the stroke.
* **Why use `imshow()`?**
  `matplotlib.pyplot.imshow()` takes a 2D grid of numerical values and renders it as an image, automatically mapping pixel intensities to colors based on a selected colormap (e.g., `binary`, `Purples`, `plasma`). This makes raw arrays human-readable.

---

## 8. Future Work

To complete the end-to-end automated grading assistant, future milestones will implement:
1. **Pixel Normalization**: Scaling raw features from `[0, 255]` to `[0, 1]` to prepare data for model training.
2. **Train-Test Splitting**: Partitioning data into subsets to validate performance on unseen digits.
3. **SVM Model Training**: Training a Support Vector Machine classifier to recognize handwritten digits.
4. **Evaluation and Visualization**: Analyzing predictions using performance metrics and visual error matrices.

---

## 9. Deliverables & Verification Results
All code and visual cells have been generated and saved inside:
* **Jupyter Notebook**: [01_data_loading.ipynb](file:///Users/rashmianand/Desktop/EdTech-Grading-Assistant/notebooks/01_data_loading.ipynb)
* The notebook is pre-run, showing clean figures for single-digit grids, multi-class sample sets, and the pixel histogram.
* No errors or warnings are present in the final run.
