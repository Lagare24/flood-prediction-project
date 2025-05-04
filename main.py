import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/predictions', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)


# Function to load data from CSV file
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Map descriptive flood status labels to numeric values for training
    flood_status_map = {
        "Very Low": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }
    df['Flood Status'] = df['Flood Status'].map(flood_status_map)
    
    # Extract features and target
    X = df[['Rainfall', 'Water Level', 'Elevation', 'Slope', 'Distance from River']]
    y = df['Flood Status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df


# Function to perform hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Function to implement Fuzzy Technique
def implement_fuzzy_system(X_train, y_train, X_test):
    # Define fuzzy variables with appropriate ranges based on the data
    rainfall = ctrl.Antecedent(np.arange(0, 100, 1), 'rainfall')
    water_level = ctrl.Antecedent(np.arange(0, 5000, 1), 'water_level')
    elevation = ctrl.Antecedent(np.arange(0, 200, 1), 'elevation')
    slope = ctrl.Antecedent(np.arange(0, 30, 1), 'slope')
    distance = ctrl.Antecedent(np.arange(0, 500, 1), 'distance')
    flood_risk = ctrl.Consequent(np.arange(0, 5, 1), 'flood_risk')  # 5 classes: 0-4

    # Define membership functions based on data distribution
    # Rainfall (in mm)
    rainfall['very_low'] = fuzz.trapmf(rainfall.universe, [0, 0, 15, 30])
    rainfall['low'] = fuzz.trapmf(rainfall.universe, [25, 35, 45, 55])
    rainfall['medium'] = fuzz.trapmf(rainfall.universe, [50, 60, 70, 80])
    rainfall['high'] = fuzz.trapmf(rainfall.universe, [75, 85, 90, 95])
    rainfall['very_high'] = fuzz.trapmf(rainfall.universe, [90, 95, 100, 100])

    # Water Level (in cm)
    water_level['very_low'] = fuzz.trapmf(water_level.universe, [0, 0, 750, 1500])
    water_level['low'] = fuzz.trapmf(water_level.universe, [1250, 1750, 2250, 2750])
    water_level['medium'] = fuzz.trapmf(water_level.universe, [2500, 3000, 3500, 4000])
    water_level['high'] = fuzz.trapmf(water_level.universe, [3750, 4250, 4500, 4750])
    water_level['very_high'] = fuzz.trapmf(water_level.universe, [4500, 4750, 5000, 5000])

    # Elevation (in meters)
    elevation['very_low'] = fuzz.trapmf(elevation.universe, [0, 0, 30, 60])
    elevation['low'] = fuzz.trapmf(elevation.universe, [50, 70, 90, 110])
    elevation['medium'] = fuzz.trapmf(elevation.universe, [100, 120, 140, 160])
    elevation['high'] = fuzz.trapmf(elevation.universe, [150, 170, 180, 190])
    elevation['very_high'] = fuzz.trapmf(elevation.universe, [180, 190, 200, 200])

    # Slope (in degrees)
    slope['very_low'] = fuzz.trapmf(slope.universe, [0, 0, 4, 8])
    slope['low'] = fuzz.trapmf(slope.universe, [6, 10, 14, 18])
    slope['medium'] = fuzz.trapmf(slope.universe, [16, 20, 22, 24])
    slope['high'] = fuzz.trapmf(slope.universe, [22, 25, 27, 29])
    slope['very_high'] = fuzz.trapmf(slope.universe, [28, 29, 30, 30])

    # Distance (in meters)
    distance['very_low'] = fuzz.trapmf(distance.universe, [0, 0, 75, 150])
    distance['low'] = fuzz.trapmf(distance.universe, [125, 175, 225, 275])
    distance['medium'] = fuzz.trapmf(distance.universe, [250, 300, 350, 400])
    distance['high'] = fuzz.trapmf(distance.universe, [375, 425, 450, 475])
    distance['very_high'] = fuzz.trapmf(distance.universe, [450, 475, 500, 500])

    # Flood Risk (output)
    flood_risk['very_low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 0.5, 1])
    flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0.5, 1, 1.5, 2])
    flood_risk['medium'] = fuzz.trapmf(flood_risk.universe, [1.5, 2, 2.5, 3])
    flood_risk['high'] = fuzz.trapmf(flood_risk.universe, [2.5, 3, 3.5, 4])
    flood_risk['very_high'] = fuzz.trapmf(flood_risk.universe, [3.5, 4, 4.5, 4.5])

    # Define comprehensive rules with weighted combinations
    rules = [
        # Very High Risk Rules
        ctrl.Rule(rainfall['very_high'] & water_level['very_high'], flood_risk['very_high']),
        ctrl.Rule(rainfall['very_high'] & water_level['high'] & (elevation['very_low'] | elevation['low']), flood_risk['very_high']),
        ctrl.Rule(rainfall['high'] & water_level['very_high'] & (elevation['very_low'] | elevation['low']), flood_risk['very_high']),
        ctrl.Rule(rainfall['very_high'] & water_level['high'] & (slope['very_low'] | slope['low']), flood_risk['very_high']),
        ctrl.Rule(rainfall['high'] & water_level['very_high'] & (distance['very_low'] | distance['low']), flood_risk['very_high']),

        # High Risk Rules
        ctrl.Rule(rainfall['high'] & water_level['high'], flood_risk['high']),
        ctrl.Rule(rainfall['high'] & water_level['medium'] & elevation['low'], flood_risk['high']),
        ctrl.Rule(rainfall['medium'] & water_level['high'] & slope['low'], flood_risk['high']),
        ctrl.Rule(rainfall['high'] & water_level['medium'] & distance['low'], flood_risk['high']),
        ctrl.Rule(rainfall['high'] & water_level['medium'] & elevation['medium'], flood_risk['high']),

        # Medium Risk Rules
        ctrl.Rule(rainfall['medium'] & water_level['medium'], flood_risk['medium']),
        ctrl.Rule(rainfall['medium'] & water_level['low'] & elevation['medium'], flood_risk['medium']),
        ctrl.Rule(rainfall['low'] & water_level['medium'] & slope['medium'], flood_risk['medium']),
        ctrl.Rule(rainfall['medium'] & water_level['low'] & distance['medium'], flood_risk['medium']),
        ctrl.Rule(rainfall['medium'] & water_level['medium'] & elevation['medium'], flood_risk['medium']),

        # Low Risk Rules
        ctrl.Rule(rainfall['low'] & water_level['low'], flood_risk['low']),
        ctrl.Rule(rainfall['low'] & water_level['very_low'] & elevation['high'], flood_risk['low']),
        ctrl.Rule(rainfall['very_low'] & water_level['low'] & slope['high'], flood_risk['low']),
        ctrl.Rule(rainfall['low'] & water_level['very_low'] & distance['high'], flood_risk['low']),
        ctrl.Rule(rainfall['low'] & water_level['low'] & elevation['high'], flood_risk['low']),

        # Very Low Risk Rules
        ctrl.Rule(rainfall['very_low'] & water_level['very_low'], flood_risk['very_low']),
        ctrl.Rule(rainfall['very_low'] & water_level['very_low'] & elevation['very_high'], flood_risk['very_low']),
        ctrl.Rule(rainfall['very_low'] & water_level['very_low'] & slope['very_high'], flood_risk['very_low']),
        ctrl.Rule(rainfall['very_low'] & water_level['very_low'] & distance['very_high'], flood_risk['very_low']),
        ctrl.Rule(rainfall['very_low'] & water_level['very_low'] & elevation['high'], flood_risk['very_low'])
    ]

    # Create control system
    flood_ctrl = ctrl.ControlSystem(rules)
    flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)

    # Make predictions
    y_pred = []
    y_pred_proba = []
    for i in range(len(X_test)):
        try:
            # Input the values
            flood_sim.input['rainfall'] = X_test[i, 0]
            flood_sim.input['water_level'] = X_test[i, 1]
            flood_sim.input['elevation'] = X_test[i, 2]
            flood_sim.input['slope'] = X_test[i, 3]
            flood_sim.input['distance'] = X_test[i, 4]
            
            # Compute the output
            flood_sim.compute()
            
            # Get the crisp output and round to nearest integer
            risk_value = flood_sim.output['flood_risk']
            y_pred.append(int(round(risk_value)))
            
            # Calculate probability-like scores using membership functions
            probs = np.zeros(5)
            for j in range(5):
                if j == 0:  # Very Low
                    probs[j] = fuzz.interp_membership(flood_risk.universe, flood_risk['very_low'].mf, risk_value)
                elif j == 1:  # Low
                    probs[j] = fuzz.interp_membership(flood_risk.universe, flood_risk['low'].mf, risk_value)
                elif j == 2:  # Medium
                    probs[j] = fuzz.interp_membership(flood_risk.universe, flood_risk['medium'].mf, risk_value)
                elif j == 3:  # High
                    probs[j] = fuzz.interp_membership(flood_risk.universe, flood_risk['high'].mf, risk_value)
                else:  # Very High
                    probs[j] = fuzz.interp_membership(flood_risk.universe, flood_risk['very_high'].mf, risk_value)
            
            # Normalize probabilities
            probs = probs / np.sum(probs)
            y_pred_proba.append(probs)
            
        except Exception as e:
            print(f"Error in fuzzy inference for sample {i}: {str(e)}")
            y_pred.append(2)  # Default to medium risk
            y_pred_proba.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))  # Equal probabilities

    # Save the fuzzy control system
    fuzzy_system = {
        'control_system': flood_ctrl,
        'antecedents': {
            'rainfall': rainfall,
            'water_level': water_level,
            'elevation': elevation,
            'slope': slope,
            'distance': distance
        },
        'consequent': flood_risk
    }
    joblib.dump(fuzzy_system, 'output/models/fuzzy_technique.joblib')

    return np.array(y_pred), np.array(y_pred_proba)


# Function to calculate entropy weights
def calculate_entropy_weights(X):
    # Normalize the data
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Calculate entropy
    entropy = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        # Add small noise to create variation if all values are the same
        if np.all(X_norm[:, j] == X_norm[0, j]):
            X_norm[:, j] += np.random.normal(0, 1e-6, size=len(X_norm))
            X_norm[:, j] = (X_norm[:, j] - X_norm[:, j].min()) / (X_norm[:, j].max() - X_norm[:, j].min())
        
        unique_values, counts = np.unique(X_norm[:, j], return_counts=True)
        probabilities = counts / len(X_norm)
        entropy[j] = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    # Calculate weights
    weights = (1 - entropy) / np.sum(1 - entropy)
    
    return weights


# Function to implement MCDA
def implement_mcda(X_train, y_train, X_test):
    # Calculate weights using entropy method
    weights = calculate_entropy_weights(X_train)
    
    # Normalize test data using min-max scaling
    X_test_norm = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
    
    # Calculate weighted sum scores
    scores = np.dot(X_test_norm, weights)
    
    # Convert scores to flood risk levels (5 classes)
    y_pred = np.zeros(len(X_test), dtype=int)
    y_pred_proba = np.zeros((len(X_test), 5))  # 5 classes
    
    # Define score ranges for each class using percentiles
    percentiles = [0, 20, 40, 60, 80, 100]
    score_ranges = [(np.percentile(scores, p1), np.percentile(scores, p2)) 
                   for p1, p2 in zip(percentiles[:-1], percentiles[1:])]
    
    for i, score in enumerate(scores):
        # Assign class based on score range
        for j, (low, high) in enumerate(score_ranges):
            if low <= score <= high:
                y_pred[i] = j
                break
        
        # Calculate probabilities using Gaussian distribution
        for j, (low, high) in enumerate(score_ranges):
            center = (low + high) / 2
            std = (high - low) / 4  # Standard deviation based on range
            y_pred_proba[i, j] = np.exp(-0.5 * ((score - center) / std) ** 2)
        
        # Normalize probabilities
        y_pred_proba[i] = y_pred_proba[i] / np.sum(y_pred_proba[i])
    
    # Save MCDA weights
    joblib.dump(weights, 'output/models/mcda_weights.joblib')
    
    return y_pred, y_pred_proba


# Function to implement AHP
def implement_ahp(X_train, y_train, X_test):
    # Define pairwise comparison matrix based on expert knowledge
    # Using Saaty's scale (1-9) for comparisons
    comparison_matrix = np.array([
        [1.0, 2.0, 4.0, 6.0, 8.0],  # Rainfall
        [1/2.0, 1.0, 3.0, 5.0, 7.0],  # Water Level
        [1/4.0, 1/3.0, 1.0, 3.0, 5.0],  # Elevation
        [1/6.0, 1/5.0, 1/3.0, 1.0, 3.0],  # Slope
        [1/8.0, 1/7.0, 1/5.0, 1/3.0, 1.0]  # Distance
    ])
    
    # Calculate weights using eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_index = np.argmax(np.real(eigenvalues))
    weights = np.real(eigenvectors[:, max_index])
    weights = weights / np.sum(weights)  # Normalize weights
    
    # Calculate consistency ratio
    n = len(comparison_matrix)
    CI = (np.real(eigenvalues[max_index]) - n) / (n - 1)
    RI = 1.12  # Random Index for n=5
    CR = CI / RI
    
    if CR > 0.1:
        print(f"Warning: AHP consistency ratio (CR={CR:.3f}) is above 0.1. The pairwise comparisons may be inconsistent.")
    
    # Normalize test data using min-max scaling
    X_test_norm = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
    
    # Calculate weighted sum scores
    scores = np.dot(X_test_norm, weights)
    
    # Convert scores to flood risk levels (5 classes)
    y_pred = np.zeros(len(X_test), dtype=int)
    y_pred_proba = np.zeros((len(X_test), 5))  # 5 classes
    
    # Define score ranges for each class using percentiles
    percentiles = [0, 20, 40, 60, 80, 100]
    score_ranges = [(np.percentile(scores, p1), np.percentile(scores, p2)) 
                   for p1, p2 in zip(percentiles[:-1], percentiles[1:])]
    
    for i, score in enumerate(scores):
        # Assign class based on score range
        for j, (low, high) in enumerate(score_ranges):
            if low <= score <= high:
                y_pred[i] = j
                break
        
        # Calculate probabilities using Gaussian distribution
        for j, (low, high) in enumerate(score_ranges):
            center = (low + high) / 2
            std = (high - low) / 4  # Standard deviation based on range
            y_pred_proba[i, j] = np.exp(-0.5 * ((score - center) / std) ** 2)
        
        # Normalize probabilities
        y_pred_proba[i] = y_pred_proba[i] / np.sum(y_pred_proba[i])
    
    # Save AHP weights and consistency ratio
    results = {
        'weights': weights,
        'consistency_ratio': CR,
        'comparison_matrix': comparison_matrix
    }
    joblib.dump(results, 'output/models/ahp_weights.joblib')
    
    return y_pred, y_pred_proba


# Function to save model and predictions
def save_model_and_predictions(model, model_name, X_test, y_test, y_pred, y_pred_proba):
    # Map numeric predictions back to descriptive labels
    flood_status_map_reverse = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }
    y_test_labels = pd.Series(y_test).map(flood_status_map_reverse)
    y_pred_labels = pd.Series(y_pred).map(flood_status_map_reverse)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'Actual': y_test_labels.reset_index(drop=True),
        'Predicted': y_pred_labels.reset_index(drop=True)
    })
    
    # Add probability columns for each class if available
    if y_pred_proba is not None:
        n_classes = y_pred_proba.shape[1]
        for i in range(n_classes):
            predictions_df[f'Probability_Class_{i}'] = y_pred_proba[:, i]
    
    # Save predictions to CSV
    predictions_df.to_csv(f'output/predictions/{model_name.lower().replace(" ", "_")}_predictions.csv', index=False)


# Function to train and evaluate model
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    print(f"\nStarted training {model_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if model_name in ['Fuzzy Technique', 'MCDA', 'AHP']:
        # Custom implementations
        y_pred, y_pred_proba = implement_fuzzy_system(X_train, y_train, X_test) if model_name == 'Fuzzy Technique' else \
                              implement_mcda(X_train, y_train, X_test) if model_name == 'MCDA' else \
                              implement_ahp(X_train, y_train, X_test)
        fpr, tpr, roc_auc = compute_roc_auc(y_test, y_pred_proba)
    else:
        # Train scikit-learn model
        model.fit(X_train, y_train)
        
        # Save the trained model
        joblib.dump(model, f'output/models/{model_name.lower().replace(" ", "_")}.joblib')
        
        # Make predictions
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)
            # Compute ROC curves and AUC
            fpr, tpr, roc_auc = compute_roc_auc(y_test, y_pred_proba)
        except (AttributeError, ValueError):
            y_pred_proba = None
            fpr, tpr, roc_auc = None, None, None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # Save predictions
    save_model_and_predictions(model, model_name, X_test, y_test, y_pred, y_pred_proba)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Finished training {model_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'training_time': training_time
    }


# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label (0=Low, 1=Medium, 2=High)')
    plt.xlabel('Predicted label (0=Low, 1=Medium, 2=High)')
    plt.savefig(f'output/plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()


# Function to plot feature importance
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'output/plots/feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()


def compute_roc_auc(y_test, y_pred_proba):
    # Binarize the labels for multiclass ROC computation
    n_classes = y_pred_proba.shape[1]
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Return the micro-average ROC curve and AUC
    return fpr["micro"], tpr["micro"], roc_auc["micro"]


# Main function
def main():
    total_start_time = time.time()
    print(f"\nStarting flood prediction model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading and preprocessing data...")
    csv_path = "data/THESIS - GIS DATA - FLOOD SCENARIOS_UPDATED.csv"
    df = load_data(csv_path)

    # Preprocess data
    X, y, df = preprocess_data(df)
    feature_names = ['Rainfall', 'Water Level', 'Elevation', 'Slope', 'Distance from River']

    # Print data summary
    print("\nData Summary:")
    print("-" * 50)
    print(f"Total samples: {len(df)}")
    flood_counts = df['Flood Status'].value_counts().sort_index()
    flood_status_map_reverse = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }
    for status, count in flood_counts.items():
        risk_level = flood_status_map_reverse[status]
        print(f"{risk_level}: {count} samples ({count / len(df) * 100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their hyperparameters
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(multi_class='ovr', max_iter=1000),
            'param_grid': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        'ANN': {
            'model': MLPClassifier(max_iter=1000),
            'param_grid': {
                'hidden_layer_sizes': [(100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2]
            }
        },
        'Fuzzy Technique': {
            'model': None,  # Custom implementation using skfuzzy
            'param_grid': {}  # No hyperparameters to tune
        },
        'MCDA': {
            'model': None,  # Custom implementation using entropy-based weighting
            'param_grid': {}  # No hyperparameters to tune
        },
        'AHP': {
            'model': None,  # Custom implementation using pairwise comparisons
            'param_grid': {}  # No hyperparameters to tune
        }
    }

    # Train and evaluate models
    results = []
    for name, config in models_config.items():
        print(f"\nTraining {name}...")

        if name in ['Fuzzy Technique', 'MCDA', 'AHP']:
            # For custom implementations
            result = train_evaluate_model(None, X_train, X_test, y_train, y_test, name)
        else:
            # For scikit-learn models
            best_model, best_params = tune_hyperparameters(
                config['model'],
                config['param_grid'],
                X_train,
                y_train
            )
            print(f"Best parameters for {name}: {best_params}")
            result = train_evaluate_model(best_model, X_train, X_test, y_train, y_test, name)

        results.append(result)
        plot_confusion_matrix(result['confusion_matrix'], name)

        if name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            plot_feature_importance(result['model'], feature_names, name)

    # Save model performance summary
    performance_df = pd.DataFrame([{
        'Model': result['model_name'],
        'Accuracy': result['accuracy'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1': result['f1'],
        'AUC': result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
    } for result in results])
    performance_df.to_csv('output/model_performance_summary.csv', index=False)
    
    # Save ROC curves
    plt.figure(figsize=(12, 8))
    for result in results:
        if result['fpr'] is not None and result['tpr'] is not None:
            plt.plot(result['fpr'], result['tpr'],
                    label=f"{result['model_name']} (AUC = {result['roc_auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig('output/plots/roc_curves.png')
    plt.close()

    # Print results
    print("\nModel Performance Summary:")
    print("-" * 100)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'Time (s)':<10}")
    print("-" * 110)
    for result in results:
        auc_value = result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
        auc_str = f"{auc_value:.4f}" if isinstance(auc_value, float) else auc_value
        print(f"{result['model_name']:<20} {result['accuracy']:.4f} {result['precision']:.4f} "
              f"{result['recall']:.4f} {result['f1']:.4f} {auc_str:<10} {result['training_time']:.2f}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
