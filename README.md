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

#### Interactive Testing (`test_models.py`)
Run:
   python test_models.py
- Provides an interactive interface for testing models
- Options to use example values or enter custom values
- Shows predictions from all models for comparison

#### Batch Testing (`test_predictions.py`)
Run:
   python test_predictions.py
- Tests models against multiple scenarios
- Allows testing with specific or random records from dataset
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

| Model             | Accuracy | Precision | Recall | F1 Score | AUC     |
| ----------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest     | 99.94%   | 100%      | 99.77% | 99.88%   | 100%    |
| Gradient Boosting | 99.94%   | 100%      | 99.77% | 99.88%   | 99.89%  |
| SVM               | 99.77%   | 100%      | 99.08% | 99.54%   | 99.99%  |
| ANN               | 99.66%   | 99.54%    | 99.08% | 99.31%   | 99.99%  |
| XGBoost           | 99.60%   | 99.54%    | 98.85% | 99.19%   | 99.99%  |
| KNN               | 90.78%   | 87.60%    | 73.10% | 79.70%   | 92.85%  |
| Fuzzy Technique   | 75.24%   | 0%        | 0%     | 0%       | 99.09%  |
| MCDA              | 68.87%   | 23.33%    | 11.26% | 15.19%   | 50.86%  |
| AHP               | 56.40%   | 32.74%    | 72.18% | 45.05%   | 65.64%  |

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
   - Prediction (Flood/No Flood)
   - Probability of flood

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
   - Consistently highest performance with 99.94% accuracy for RF/GB and 99.60% for XGBoost
   - Perfect precision (100%) indicates no false positives - crucial for flood warnings
   - High recall (>98.85%) shows excellent ability to detect actual flood events
   - Feature importance analysis shows:
     - Water level is the most critical predictor
     - Rainfall rate and elevation are secondary but significant factors
     - Distance from river and slope have moderate influence

2. **Support Vector Machine (SVM)**
   - Excellent performance (99.77% accuracy) with linear kernel
   - Perfect precision but slightly lower recall than tree-based models
   - Very high AUC (99.99%) indicates excellent discrimination ability
   - Works well due to clear separation between flood/non-flood cases

3. **Artificial Neural Network (ANN)**
   - Strong performance (99.66% accuracy) with single hidden layer
   - Balanced precision and recall (both 99.54%)
   - Performs well with both linear and non-linear relationships
   - Optimal with tanh activation and adam optimizer

4. **K-Nearest Neighbors (KNN)**
   - Moderate performance (90.78% accuracy)
   - Lower recall (73.10%) indicates missed flood events
   - Works best with k=3 and distance-weighted voting
   - Performance suggests some overlap in feature space

### Alternative Approaches

5. **Fuzzy Logic System**
   - Lower traditional metrics but high AUC (99.09%)
   - Uses linguistic rules for interpretability
   - Good for handling uncertainty in measurements
   - Three-level classification (low/medium/high) for each input

6. **Multi-Criteria Decision Analysis (MCDA)**
   - Moderate performance (68.87% accuracy)
   - Uses entropy-weighted criteria
   - Transparent decision process
   - Suitable for stakeholder involvement

7. **Analytic Hierarchy Process (AHP)**
   - Basic performance (56.40% accuracy)
   - Higher recall (72.18%) but lower precision
   - Structured approach to criteria weighting
   - Good for incorporating expert knowledge

### ROC Curve Analysis
- All traditional ML models show excellent ROC curves (AUC > 0.99)
- Clear separation between flood and non-flood cases
- Random Forest achieves perfect AUC (1.0)
- Alternative approaches show more varied discrimination ability

## Notes

- The best performing models are Random Forest, Gradient Boosting, and SVM
- Some models (Fuzzy, MCDA, AHP) have lower accuracy but provide better interpretability
- All models were trained on the same dataset with consistent preprocessing
- Model selection should consider both performance and interpretability needs
- Ensemble methods (RF, GB, XGBoost) provide the most robust predictions
