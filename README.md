# Flood Prediction Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements a comprehensive flood prediction system that combines multiple approaches to assess flood risk based on environmental factors. The system integrates:

### 1. Machine Learning Models

- **Tree-based Models**: XGBoost, Random Forest, Gradient Boosting
- **Neural Networks**: Multi-layer Perceptron (ANN)
- **Traditional ML**: SVM, KNN, Logistic Regression

### 2. Decision Support Systems

- **Fuzzy Logic**: Handles uncertainty in environmental measurements with refined membership functions and rules.
- **Multi-Criteria Decision Analysis (MCDA)**: Balances multiple risk factors using entropy-based weights.
- **Analytic Hierarchy Process (AHP)**: Structured expert knowledge integration with validated pairwise comparison matrices.

### 3. Key Features

- Real-time flood risk assessment
- Multi-level risk classification (Very Low, Low, Medium, High, Very High)
- Comparative model analysis
- Interactive testing interface
- Detailed performance metrics
- Visualization tools (e.g., feature importance, ROC curves, confusion matrices)

### 4. Applications

- Early warning systems
- Urban planning
- Disaster preparedness
- Risk management
- Policy decision support

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Lagare24/flood-prediction-project.git
cd flood-prediction-project
```

2. Create and activate a virtual environment (recommended):

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For Linux/MacOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

### Updating the Project

To get the latest changes:

1. Save any local changes:

```bash
git add .
git commit -m "Save local changes"
```

2. Fetch updates from remote:

```bash
git fetch origin
# View changes
git log --oneline HEAD..origin/main
```

3. Pull the changes:

```bash
git pull origin main
```

4. If there are conflicts:
   - Git will mark the conflicting files
   - Resolve conflicts in your editor
   - Commit the resolved changes:

```bash
git add .
git commit -m "Resolve merge conflicts"
```

5. Update dependencies if needed:

```bash
pip install -r requirements.txt
```

Note: If you haven't made any local changes, you can simply run:

```bash
git fetch origin
git pull origin main
```

## Project Structure

```
flood-prediction-project/
├── data/                  # Dataset directory
│   └── THESIS - GIS DATA - FLOOD SCENARIOS.csv  # Main dataset
├── output/               # Generated after training
│   ├── models/          # Saved model files
│   ├── predictions/     # Model predictions
│   └── plots/          # Performance visualizations
├── main.py              # Model training script
├── test_models.py       # Interactive testing script
├── test_predictions.py  # Batch prediction script
├── update_flood_status.py  # Risk scoring system
└── requirements.txt     # Package dependencies
```

### Key Files Description

1. **main.py**

   - Core script for training all models
   - Implements grid search for hyperparameter tuning
   - Generates performance metrics and visualizations
   - Saves trained models and predictions

2. **test_models.py**

   - Interactive testing interface
   - Supports both example and custom input values
   - Shows predictions from all models side by side
   - Includes probability scores for each risk level

3. **test_predictions.py**

   - Batch prediction interface
   - Detailed reporting with confidence scores
   - Supports CSV input for multiple scenarios
   - Generates comprehensive prediction reports

4. **update_flood_status.py**
   - Implements a rule-based scoring system
   - Evaluates flood risk based on 5 key parameters:
     - Rainfall Rate (0-100 mm/hr)
     - Water Level (0-5000 mm)
     - Elevation (0-1000 m)
     - Slope (0-90 degrees)
     - Distance from River (0-5000 m)
   - Uses weighted scoring for each parameter
   - Classifies risk into three levels:
     - High Risk (avg score ≥ 4)
     - Medium Risk (3 ≤ avg score < 4)
     - Low Risk (avg score < 3)
   - Updates the main dataset with calculated risk levels

## Usage

### 1. Training Models

To train all models and generate performance metrics, run:

```bash
python main.py
```

This will:

- Load and preprocess the dataset
- Perform train-test split (80-20)
- Train all models using grid search for optimal parameters
- Generate performance metrics (accuracy, precision, recall, F1, AUC)
- Create visualization plots:
  - Confusion matrices
  - ROC curves
  - Feature importance (for tree-based models)
- Save trained models in `output/models/`
- Save predictions in `output/predictions/`

Training time varies by model:

- Fast (< 1 min): Logistic Regression, KNN, AHP, MCDA
- Medium (1-5 min): SVM, Random Forest, Gradient Boosting
- Slower (5+ min): XGBoost, ANN (depends on convergence)

### 2. Testing Model

Three testing approaches are provided:

#### B. Batch Prediction Testing (`test_predictions.py`)

```bash
python test_predictions.py
```

Features:

- Handles multiple scenarios at once
- Supports CSV input files
- Generates detailed reports including:
  - Prediction summaries
  - Confidence metrics
  - Model agreement analysis
  - Risk level distribution
- Export options:
  - CSV format for further analysis
  - Summary statistics
  - Comparison charts

#### C. Risk Scoring System (`update_flood_status.py`)

```bash
python update_flood_status.py
```

Features:

- Rule-based scoring system
- Parameter-specific risk assessment:
  - Rainfall intensity scoring
  - Water level thresholds
  - Elevation risk zones
  - Slope impact analysis
  - Distance-based vulnerability
- Weighted risk calculation
- Automated dataset updates
- Summary statistics output

## Output Files

### 1. Models

Trained models are saved in `output/models/`:

- `svm.joblib`: Support Vector Machine (SVM) model
- `random_forest.joblib`: Random Forest model
- `knn.joblib`: K-Nearest Neighbors (KNN) model
- `ann.joblib`: Artificial Neural Network (ANN) model
- `xgboost.joblib`: XGBoost model
- `gradient_boosting.joblib`: Gradient Boosting model
- `logistic_regression.joblib`: Logistic Regression model
- `fuzzy_technique.joblib`: Fuzzy Logic model (handles uncertainty with refined membership functions and rules)
- `mcda_weights.joblib`: Multi-Criteria Decision Analysis (MCDA) model (uses entropy-based weights for decision-making)
- `ahp_weights.joblib`: Analytic Hierarchy Process (AHP) model (uses pairwise comparison matrices for structured decision-making)

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

| Feature             | Description                   | Unit    | Range  |
| ------------------- | ----------------------------- | ------- | ------ |
| Rainfall Rate       | Rate of rainfall              | mm/hr   | 0-100  |
| Water level         | Water level in the river      | mm      | 0-5000 |
| elevation           | Ground elevation at the point | m       | 0-1000 |
| Slope               | Ground slope                  | degrees | 0-90   |
| Distance from River | Distance from nearest river   | m       | 0-5000 |

### Data Format

Input data should be provided as a CSV file with the following structure:

```csv
Rainfall,Water Level,Elevation,Slope,Distance from River
19.93,4399.39,8.3,5.59,322.0
```

### Flood Status Values

The flood risk is encoded as integer values:

- **0**: Low Risk

  - Areas with minimal flooding probability
  - Good drainage and elevated terrain
  - Low historical flood incidents

- **1**: Medium Risk

  - Moderate chance of flooding
  - Some vulnerability to heavy rainfall
  - Occasional historical flooding

- **2**: High Risk
  - High probability of flooding
  - Poor drainage or low-lying areas
  - Frequent historical flooding

These values are used consistently across:

- Training data labels
- Model predictions
- Risk scoring system
- Performance evaluations

The risk levels are determined by:

1. **Rule-based Scoring** (update_flood_status.py):

   - Calculates weighted scores for each parameter
   - Averages the scores (range 1-5)
   - Maps to risk levels:
     - High (2): avg_score ≥ 4
     - Medium (1): 3 ≤ avg_score < 4
     - Low (0): avg_score < 3

2. **Historical Data**:
   - Past flood events
   - Damage assessments
   - Expert classifications

Notes:

- DATE format: DD/MM/YYYY
- Missing values should be marked as empty or NaN
- The system will automatically handle missing values using forward fill
- Water level threshold for flood classification is set at the 75th percentile

## Model Performance Summary

### Performance Metrics

| Model               | Accuracy | Precision | Recall | F1 Score | AUC    | Training Time\* |
| ------------------- | -------- | --------- | ------ | -------- | ------ | --------------- |
| XGBoost             | 99.98%   | 99.99%    | 99.93% | 99.96%   | 100%   | 5-10 min        |
| Random Forest       | 99.84%   | 99.79%    | 99.44% | 99.61%   | 100%   | 1-3 min         |
| Gradient Boosting   | 99.95%   | 99.97%    | 99.80% | 99.88%   | 100%   | 2-4 min         |
| ANN                 | 99.01%   | 98.65%    | 98.17% | 98.41%   | 99.93% | 5-15 min        |
| KNN                 | 93.01%   | 92.00%    | 91.89% | 91.94%   | 98.27% | < 1 min         |
| SVM                 | 87.80%   | 87.41%    | 81.76% | 84.18%   | 97.03% | 2-5 min         |
| Logistic Regression | 66.78%   | 65.92%    | 64.31% | 65.10%   | 81.23% | < 1 min         |
| Fuzzy Technique     | 56.13%   | 55.87%    | 54.92% | 55.39%   | N/A    | < 1 min         |
| MCDA                | 30.47%   | 30.47%    | 27.38% | 26.43%   | N/A    | < 1 min         |
| AHP                 | 42.32%   | 42.32%    | 44.35% | 38.94%   | N/A    | < 1 min         |

\*Training times are approximate and may vary based on hardware

### Model Selection Guide

#### High Accuracy Requirements

1. **XGBoost** (99.98% accuracy)

   - Best overall performance
   - Excellent for real-time predictions
   - Handles complex feature relationships
   - Resource-intensive training

2. **Gradient Boosting** (99.95% accuracy)

   - Very close to XGBoost performance
   - More memory efficient
   - Faster training time
   - Good for production deployment

3. **Random Forest** (99.84% accuracy)
   - Highly reliable and stable
   - Easy to tune and maintain
   - Built-in feature importance
   - Excellent for baseline modeling

#### Balanced Performance

4. **ANN** (99.01% accuracy)

   - Great for complex patterns
   - Handles non-linear relationships
   - Requires more training data
   - Good for automated systems

5. **KNN** (93.01% accuracy)

   - Simple and interpretable
   - Fast predictions
   - Good for small-medium datasets
   - Works well with normalized data

6. **SVM** (87.80% accuracy)
   - Robust to outliers
   - Works well with clear margins
   - Moderate training time
   - Good for binary decisions

#### Interpretability Focus

7. **Logistic Regression** (66.78% accuracy)

   - Very interpretable
   - Fast training and prediction
   - Good for understanding feature impacts
   - Baseline model for comparison

8. **Fuzzy Technique** (56.13% accuracy)

   - Handles uncertainty well
   - Human-readable rules
   - Good for expert systems
   - Matches human reasoning

9. **AHP** (42.32% accuracy)

   - Structured decision process
   - Incorporates expert knowledge
   - Clear criteria weights
   - Good for policy making

10. **MCDA** (30.47% accuracy) - Multi-criteria evaluation - Transparent methodology - Stakeholder involvement - Policy decision support
    | Logistic Regression | 66.78% | 78.65% | 46.69% | 46.60% | 81.67% |
    | Fuzzy Technique | 56.13% | 18.71% | 33.33% | 23.97% | N/A |
    | MCDA | 30.47% | 30.47% | 27.38% | 26.43% | N/A |
    | AHP | 42.32% | 42.32% | 44.35% | 38.94% | N/A |

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
