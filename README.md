# Breast Cancer Classification using Artificial Neural Networks

## Overview

This project implements a binary classification model to predict breast cancer diagnosis using an artificial neural network (ANN). The dataset used is loaded from a CSV file and contains features extracted from breast cancer tumors.

## Prerequisites

Ensure you have the following Python packages installed before running the code:

```sh
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow keras
```

## Dataset

The dataset is loaded from:

```python
df = pd.read_csv('C:/Users/Ali Traders/Downloads/Classificaton_ANN/data.csv')
```

Make sure to replace the file path with the actual location of your dataset.

## Steps in the Project

### 1. Data Preprocessing

- Load the dataset.
- Remove unnecessary columns (`Unnamed: 32`).
- Encode the `diagnosis` column using Label Encoding.
- Split the dataset into training and testing sets.
- Standardize the feature values using `StandardScaler`.

### 2. Building the ANN Model

- The model consists of three layers:
  - Input layer with 16 neurons and ReLU activation.
  - Hidden layer with 8 neurons and ReLU activation.
  - Output layer with 1 neuron and sigmoid activation.
- The model is compiled using Adam optimizer and binary cross-entropy loss function.
- The model is trained on the training dataset for 150 epochs with a batch size of 100.

### 3. Evaluation

- The model predictions are compared with the actual values.
- A confusion matrix is plotted using Seaborn.
- Model accuracy is assessed.

### 4. K-Fold Cross Validation

- The model is evaluated using 10-fold cross-validation to measure its robustness.
- The mean and variance of accuracies are calculated.

### 5. Dropout Regularization

- Dropout layers with a 10% dropout rate are added to the model to prevent overfitting.

### 6. Hyperparameter Tuning

- The model is optimized using GridSearchCV to find the best batch size, epochs, and optimizer.

## Errors & Fixes

There were some errors in the original code that have been fixed:

- `kerasClassifier` corrected to `KerasClassifier`.
- `Saequential()` corrected to `Sequential()`.
- `p=0.1` in `Dropout` corrected to `rate=0.1`.
- `tensorflow.keras.models_selection` corrected to `sklearn.model_selection`.
- `best_param_` corrected to `best_params_`.

## Running the Code

Run the script in a Python environment (Jupyter Notebook or standalone script). Ensure the dataset path is correct before execution.

```sh
python breast_cancer_ann.py
```

## Output

- Model accuracy and loss.
- Confusion matrix visualization (`h.png`).
- Best hyperparameters from GridSearchCV.

## Author

BILAWAL\_MIR

