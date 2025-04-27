# Flood Prediction Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements multiple machine learning and decision-making approaches to predict flood events based on environmental factors. It includes traditional ML models (SVM, Random Forest, etc.) and alternative approaches (Fuzzy Logic, MCDA, AHP) for comparison.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:

   git clone https://github.com/Lagare24/flood-prediction-project.git
   cd flood-prediction-project

2. Create and activate a virtual environment (recommended):

   For Windows:
   python -m venv .venv
   .venv\Scripts\activate

   For Linux/MacOS:
   python -m venv .venv
   source .venv/bin/activate

3. Install required packages:

   pip install -r requirements.txt

## Project Structure

```
flood-prediction-project/
├── data/                  # Dataset directory
├── output/               # Generated after training
│   ├── models/          # Saved model files
│   ├── predictions/     # Model predictions
│   └── plots/          # Performance visualizations
├── main.py              # Model training script
├── test_models.py       # Interactive testing script
├── test_predictions.py  # Batch prediction script
└── requirements.txt     # Package dependencies
```

## Usage

### 1. Training Models

To train all models and generate performance metrics, run:

   python main.py

This will:
- Train all models using the dataset
- Save trained models in `output/models/`
- Generate performance plots in `output/plots/`
- Save prediction results in `output/predictions/`

### 2. Testing Models

Two testing scripts are provided:

#### Hard-coded input Testing (`test_models.py`)
Run:
   python test_models.py
- Provides static input for twsting models
- Options to use example values or enter custom values
- Shows predictions from all models for comparison

#### Interactive Testing (`test_predictions.py`)
Run:
   python test_predictions.py
- Provides an interactive interface for testing models
- Allows testing with user interaction, input value via prompt
- Generates detailed prediction reports

## Output Files

### 1. Models

Trained models are saved in `output/models/`:

- `svm.joblib`: Support Vector Machine model
- `random_forest.joblib`: Random Forest model
- `knn.joblib`: K-Nearest Neighbors model
- `ann.joblib`: Artificial Neural Network model
- `xgboost.joblib`: XGBoost model
- `gradient_boosting.joblib`: Gradient Boosting model
- `fuzzy_technique.joblib`: Fuzzy Logic model
- `mcda.joblib`: Multi-Criteria Decision Analysis model
- `ahp.joblib`: Analytic Hierarchy Process model

### 2. Predictions

Prediction results in `output/predictions/`:
- Individual CSV files for each model (e.g., `svm_predictions.csv`)
- Contains actual values, predicted values, and prediction probabilities

### 3. Performance Plots

Visualizations are saved in `output/plots`:

- Confusion matrices for each model
- Feature importance plots for tree-based models
- ROC curves showing model performance

## Input Data Format

### Features

The models expect the following input features:

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| Rainfall Rate | Rate of rainfall | mm/hr | 0-100 |
| Water level | Water level in the river | mm | 0-5000 |
| elevation | Ground elevation at the point | m | 0-1000 |
| Slope | Ground slope | degrees | 0-90 |
| Distance from River | Distance from nearest river | m | 0-5000 |

### Data Format

Input data should be provided as a CSV file with the following structure:

```csv
DATE,Rainfall Rate,Water level,elevation,Slope,Distance from River
25/04/2025,45.2,2500.0,150.3,12.5,300.0
```

Notes:
- DATE format: DD/MM/YYYY
- Missing values should be marked as empty or NaN
- The system will automatically handle missing values using forward fill
- Water level threshold for flood classification is set at the 75th percentile

## Model Performance Summary

| Model               | Accuracy | Precision | Recall  | F1 Score | AUC     |
|---------------------|----------|-----------|---------|----------|---------|
| XGBoost             | 99.98%   | 99.99%   | 99.93%  | 99.96%   | 100%    |
| Random Forest       | 99.84%   | 99.79%   | 99.44%  | 99.61%   | 100%    |
| Gradient Boosting   | 99.95%   | 99.97%   | 99.80%  | 99.88%   | 100%    |
| ANN                 | 99.01%   | 98.65%   | 98.17%  | 98.41%   | 99.93%  |
| KNN                 | 93.01%   | 92.00%   | 91.89%  | 91.94%   | 98.27%  |
| SVM                 | 87.80%   | 87.41%   | 81.76%  | 84.18%   | 97.03%  |
| Logistic Regression | 66.78%   | 78.65%   | 46.69%  | 46.60%   | 81.67%  |
| Fuzzy Technique     | 56.13%   | 18.71%   | 33.33%  | 23.97%   | N/A     |
| MCDA                | 30.47%   | 30.47%   | 27.38%  | 26.43%   | N/A     |
| AHP                 | 42.32%   | 42.32%   | 44.35%  | 38.94%   | N/A     |

## How to Use

### Testing Models

To test the models with custom input values:

1. Run the test script:

```bash
python test_models.py
```

2. The script will test all models with example input values:

```python
test_input = {
    'Rainfall Rate': 50.0,
    'Water level': 2000.0,
    'elevation': 100.0,
    'Slope': 5.0,
    'Distance from River': 500.0
}
```

3. For each model, you'll see:
   - Input values used
   - Prediction (Low/Medium/High Risk)
   - Risk probabilities or scores

### Using Models in Your Code

1. Models can be loaded using Python's joblib library:

```python
import joblib
model = joblib.load('output/models/model_name.joblib')
```

2. Each model can be used to make predictions on new data
3. The prediction files can be used to analyze model performance

## Model Interpretations

### Performance Analysis

1. **Tree-based Models (Random Forest, Gradient Boosting, XGBoost)**
   - XGBoost leads with 99.98% accuracy, followed by Gradient Boosting (99.95%) and Random Forest (99.84%)
   - Near-perfect precision (>99.79%) indicates extremely reliable risk level predictions
   - Excellent recall (>99.44%) shows outstanding ability to detect all risk levels
   - Feature importance analysis shows:
     - Water level is the most critical predictor
     - Rainfall rate and elevation are secondary but significant factors
     - Distance from river and slope have moderate influence

2. **Support Vector Machine (SVM) and Logistic Regression**
   - SVM achieves 87.80% accuracy with good balance between precision and recall
   - Logistic Regression shows moderate performance (66.78% accuracy)
   - Both models perform better than traditional methods but lag behind modern ML approaches
   - Good for interpretability and understanding feature relationships

3. **Artificial Neural Network (ANN)**
   - Excellent performance (99.01% accuracy) with optimized architecture
   - Very balanced precision (98.65%) and recall (98.17%)
   - Performs well with both linear and non-linear relationships
   - Optimal with tanh activation and adam optimizer

4. **K-Nearest Neighbors (KNN)**
   - Good performance (93.01% accuracy)
   - Well-balanced precision (92.00%) and recall (91.89%)
   - Works best with optimized k value and distance-weighted voting
   - Effective for capturing local patterns in the feature space

### Alternative Approaches

5. **Fuzzy Logic System**
   - Moderate performance (56.13% accuracy)
   - Better at capturing uncertainty in measurements
   - Uses intuitive linguistic rules for interpretability
   - Three-level classification (low/medium/high) matches human reasoning

6. **Multi-Criteria Decision Analysis (MCDA)**
   - Lower performance (30.47% accuracy)
   - Uses entropy-weighted criteria for objective weighting
   - Transparent and explainable decision process
   - Suitable for stakeholder involvement and policy making

7. **Analytic Hierarchy Process (AHP)**
   - Basic performance (42.32% accuracy)
   - Balanced precision and recall around 43%
   - Structured approach to criteria weighting
   - Good for incorporating expert knowledge and preferences

### ROC Curve Analysis
- Tree-based models achieve perfect AUC (1.0)
- ANN and KNN show excellent discrimination (AUC > 0.98)
- SVM and Logistic Regression maintain good ROC curves (AUC > 0.81)
- Traditional methods (Fuzzy, MCDA, AHP) focus on interpretability over ROC metrics

## Notes

- XGBoost, Gradient Boosting, and Random Forest provide the most accurate predictions
- ANN and KNN offer good balance between accuracy and computational efficiency
- Traditional methods (Fuzzy, MCDA, AHP) prioritize interpretability over accuracy
- All models support three-level risk classification (Low, Medium, High)
- Model selection should balance accuracy, speed, and interpretability needs
