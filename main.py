import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
def calculate_flood_status(row):
    # Initialize score
    score = 0
    
    # Rainfall scoring
    if row['Rainfall'] > 30: score += 5  # Very High
    elif 15 <= row['Rainfall'] <= 30: score += 4  # High
    elif 7.5 <= row['Rainfall'] < 15: score += 3  # Medium
    elif 2.5 <= row['Rainfall'] < 7.5: score += 2  # Low
    else: score += 1  # Very Low
    
    # Water Level scoring
    if row['Water Level'] > 4500: score += 5  # Very High
    elif 4300 <= row['Water Level'] <= 4400: score += 4  # High
    elif 4200 <= row['Water Level'] < 4300: score += 3  # Medium
    elif 4000 <= row['Water Level'] < 4200: score += 2  # Low
    else: score += 1  # Very Low
    
    # Elevation scoring
    if 0 <= row['Elevation'] <= 5: score += 5  # Very High
    elif 6 <= row['Elevation'] <= 20: score += 4  # High
    elif 21 <= row['Elevation'] <= 50: score += 3  # Medium
    elif 51 <= row['Elevation'] <= 150: score += 2  # Low
    else: score += 1  # Very Low
    
    # Slope scoring
    if 0 <= row['Slope'] <= 3: score += 5  # Very High
    elif 3 < row['Slope'] <= 8: score += 4  # High
    elif 8 < row['Slope'] <= 18: score += 3  # Medium
    elif 18 < row['Slope'] <= 30: score += 2  # Low
    else: score += 1  # Very Low
    
    # Distance from River scoring
    if 0 <= row['Distance from River'] <= 100: score += 5  # Very High
    elif 100 < row['Distance from River'] <= 200: score += 4  # High
    elif 200 < row['Distance from River'] <= 300: score += 3  # Medium
    elif 300 < row['Distance from River'] <= 400: score += 2  # Low
    else: score += 1  # Very Low
    
    # Calculate final flood status based on average score
    avg_score = score / 5  # 5 parameters
    
    if avg_score >= 4: return 2  # High flood risk
    elif avg_score >= 3: return 1  # Medium flood risk
    else: return 0  # Low flood risk

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(method='ffill')
    
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
    # Define fuzzy variables for each input
    rainfall = ctrl.Antecedent(np.arange(0, 100, 1), 'rainfall')
    water_level = ctrl.Antecedent(np.arange(0, 5000, 1), 'water_level')
    elevation = ctrl.Antecedent(np.arange(0, 200, 1), 'elevation')
    slope = ctrl.Antecedent(np.arange(0, 30, 1), 'slope')
    distance = ctrl.Antecedent(np.arange(0, 500, 1), 'distance')
    flood_risk = ctrl.Consequent(np.arange(0, 3, 1), 'flood_risk')

    # Define membership functions for rainfall
    rainfall['very_low'] = fuzz.trimf(rainfall.universe, [0, 0, 2.5])
    rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 2.5, 7.5])
    rainfall['medium'] = fuzz.trimf(rainfall.universe, [7.5, 15, 30])
    rainfall['high'] = fuzz.trimf(rainfall.universe, [15, 30, 100])
    rainfall['very_high'] = fuzz.trimf(rainfall.universe, [30, 100, 100])

    # Define membership functions for water level
    water_level['very_low'] = fuzz.trimf(water_level.universe, [0, 0, 4000])
    water_level['low'] = fuzz.trimf(water_level.universe, [4000, 4000, 4200])
    water_level['medium'] = fuzz.trimf(water_level.universe, [4200, 4300, 4300])
    water_level['high'] = fuzz.trimf(water_level.universe, [4300, 4400, 4400])
    water_level['very_high'] = fuzz.trimf(water_level.universe, [4400, 4500, 5000])

    # Define membership functions for elevation
    elevation['very_high'] = fuzz.trimf(elevation.universe, [0, 0, 5])
    elevation['high'] = fuzz.trimf(elevation.universe, [5, 10, 20])
    elevation['medium'] = fuzz.trimf(elevation.universe, [20, 35, 50])
    elevation['low'] = fuzz.trimf(elevation.universe, [50, 100, 150])
    elevation['very_low'] = fuzz.trimf(elevation.universe, [150, 200, 200])

    # Define membership functions for slope
    slope['very_high'] = fuzz.trimf(slope.universe, [0, 0, 3])
    slope['high'] = fuzz.trimf(slope.universe, [3, 5, 8])
    slope['medium'] = fuzz.trimf(slope.universe, [8, 13, 18])
    slope['low'] = fuzz.trimf(slope.universe, [18, 24, 30])
    slope['very_low'] = fuzz.trimf(slope.universe, [30, 30, 30])

    # Define membership functions for distance
    distance['very_high'] = fuzz.trimf(distance.universe, [0, 0, 100])
    distance['high'] = fuzz.trimf(distance.universe, [100, 150, 200])
    distance['medium'] = fuzz.trimf(distance.universe, [200, 250, 300])
    distance['low'] = fuzz.trimf(distance.universe, [300, 350, 400])
    distance['very_low'] = fuzz.trimf(distance.universe, [400, 500, 500])

    # Define membership functions for flood risk (0=Low, 1=Medium, 2=High)
    flood_risk['low'] = fuzz.trimf(flood_risk.universe, [0, 0, 1])
    flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [0, 1, 2])
    flood_risk['high'] = fuzz.trimf(flood_risk.universe, [1, 2, 2])

    # Define rules
    rules = [
        # High flood risk rules
        ctrl.Rule((rainfall['very_high'] | rainfall['high']) & 
                 (water_level['very_high'] | water_level['high']) & 
                 (elevation['very_high'] | elevation['high']) & 
                 (slope['very_high'] | slope['high']) & 
                 (distance['very_high'] | distance['high']), 
                 flood_risk['high']),
        
        # Medium flood risk rules
        ctrl.Rule((rainfall['medium'] | water_level['medium'] | elevation['medium'] | 
                  slope['medium'] | distance['medium']), 
                 flood_risk['medium']),
        
        # Low flood risk rules
        ctrl.Rule((rainfall['very_low'] | rainfall['low']) & 
                 (water_level['very_low'] | water_level['low']) & 
                 (elevation['very_low'] | elevation['low']) & 
                 (slope['very_low'] | slope['low']) & 
                 (distance['very_low'] | distance['low']), 
                 flood_risk['low'])
    ]

    # Create control system
    flood_ctrl = ctrl.ControlSystem(rules)
    flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)

    # Make predictions
    y_pred = []
    for i in range(len(X_test)):
        try:
            flood_sim.input['rainfall'] = X_test[i, 0]
            flood_sim.input['water_level'] = X_test[i, 1]
            flood_sim.input['elevation'] = X_test[i, 2]
            flood_sim.input['slope'] = X_test[i, 3]
            flood_sim.input['distance'] = X_test[i, 4]
            flood_sim.compute()
            
            # Convert continuous output to discrete classes
            risk_value = flood_sim.output['flood_risk']
            if risk_value < 0.5:
                y_pred.append(0)  # Low risk
            elif risk_value < 1.5:
                y_pred.append(1)  # Medium risk
            else:
                y_pred.append(2)  # High risk
        except:
            # If fuzzy system fails, predict medium risk as default
            y_pred.append(1)

    return np.array(y_pred)


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
    
    # Normalize test data
    X_test_norm = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
    
    # Calculate weighted sum scores
    scores = np.dot(X_test_norm, weights)
    
    # Convert scores to flood risk levels
    y_pred = np.zeros(len(X_test), dtype=int)
    y_pred[scores >= np.percentile(scores, 66.67)] = 2  # High risk
    y_pred[(scores >= np.percentile(scores, 33.33)) & (scores < np.percentile(scores, 66.67))] = 1  # Medium risk
    y_pred[scores < np.percentile(scores, 33.33)] = 0  # Low risk
    
    return y_pred


# Function to implement AHP
def implement_ahp(X_train, y_train, X_test):
    # Define pairwise comparison matrix (example values)
    comparison_matrix = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Rainfall
        [0.5, 1.0, 2.0, 3.0, 4.0],  # Water Level
        [0.33, 0.5, 1.0, 2.0, 3.0],  # Elevation
        [0.25, 0.33, 0.5, 1.0, 2.0], # Slope
        [0.2, 0.25, 0.33, 0.5, 1.0]  # Distance
    ])
    
    # Calculate weights using eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    weights = np.real(eigenvectors[:, 0] / np.sum(eigenvectors[:, 0]))
    
    # Normalize test data
    X_test_norm = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
    
    # Calculate weighted sum scores
    scores = np.dot(X_test_norm, weights)
    
    # Convert scores to flood risk levels
    y_pred = np.zeros(len(X_test), dtype=int)
    y_pred[scores >= np.percentile(scores, 66.67)] = 2  # High risk
    y_pred[(scores >= np.percentile(scores, 33.33)) & (scores < np.percentile(scores, 66.67))] = 1  # Medium risk
    y_pred[scores < np.percentile(scores, 33.33)] = 0  # Low risk
    
    return y_pred


# Function to save model and predictions
def save_model_and_predictions(model, model_name, X_test, y_test, y_pred, y_pred_proba):
    # Save the model
    if model is not None:
        joblib.dump(model, f'output/models/{model_name.lower().replace(" ", "_")}.joblib')
    elif model_name == 'Fuzzy Technique':
        # Save fuzzy system parameters
        fuzzy_params = {
            'rainfall_range': (0, 100),
            'water_level_range': (0, 5000),
            'elevation_range': (0, 200),
            'slope_range': (0, 30),
            'distance_range': (0, 500),
            'flood_risk_range': (0, 2),
            'membership_levels': ['very_low', 'low', 'medium', 'high', 'very_high'],
            'output_classes': ['low', 'medium', 'high']
        }
        joblib.dump(fuzzy_params, f'output/models/fuzzy_technique.joblib')
    elif model_name == 'MCDA':
        # Save MCDA parameters
        mcda_params = {
            'method': 'entropy_weighting',
            'normalization': 'minmax',
            'criteria_weights': calculate_entropy_weights(X_test).tolist()
        }
        joblib.dump(mcda_params, f'output/models/mcda.joblib')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Add probability columns for each class if available
    if y_pred_proba is not None:
        n_classes = y_pred_proba.shape[1]
        for i in range(n_classes):
            predictions_df[f'Probability_Class_{i}'] = y_pred_proba[:, i]
    
    predictions_df.to_csv(f'output/predictions/{model_name.lower().replace(" ", "_")}_predictions.csv', index=False)


# Function to train and evaluate model
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    print(f"\nStarted training {model_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if model_name == 'Fuzzy Technique':
        y_pred = implement_fuzzy_system(X_train, y_train, X_test)
        y_pred_proba = None
        fpr, tpr, roc_auc = None, None, None
    elif model_name == 'MCDA':
        y_pred = implement_mcda(X_train, y_train, X_test)
        y_pred_proba = None
        fpr, tpr, roc_auc = None, None, None
    elif model_name == 'AHP':
        y_pred = implement_ahp(X_train, y_train, X_test)
        y_pred_proba = None
        fpr, tpr, roc_auc = None, None, None
    else:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)
            # For multiclass ROC, we'll use macro averaging
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            n_classes = len(np.unique(y_test))
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute macro-average ROC curve and ROC area
            fpr['macro'] = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            tpr['macro'] = np.zeros_like(fpr['macro'])
            for i in range(n_classes):
                tpr['macro'] += np.interp(fpr['macro'], fpr[i], tpr[i])
            tpr['macro'] /= n_classes
            roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
            
            # Use macro averages for final values
            fpr = fpr['macro']
            tpr = tpr['macro']
            roc_auc = roc_auc['macro']
        except (AttributeError, ValueError):
            y_pred_proba = None
            fpr, tpr, roc_auc = None, None, None
    
    # Calculate metrics with macro averaging for multiclass
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
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


# Main function
def main():
    total_start_time = time.time()
    print(f"\nStarting flood prediction model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading and preprocessing data...")
    csv_path = "data/THESIS - GIS DATA - FLOOD SCENARIOS.csv"
    df = load_data(csv_path)

    # Preprocess data
    X, y, df = preprocess_data(df)
    feature_names = ['Rainfall', 'Water Level', 'Elevation', 'Slope', 'Distance from River']

    # Print data summary
    print("\nData Summary:")
    print("-" * 50)
    print(f"Total samples: {len(df)}")
    flood_counts = df['Flood Status'].value_counts().sort_index()
    for status, count in flood_counts.items():
        risk_level = 'Low' if status == 0 else 'Medium' if status == 1 else 'High'
        print(f"{risk_level} risk events: {count} ({count/len(df)*100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their hyperparameters
    models_config = {
        'Logistic Regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', LogisticRegression(multi_class='ovr', max_iter=1000))
            ]),
            'param_grid': {
                'estimator__C': [0.1, 1, 10],
                'estimator__penalty': ['l1', 'l2'],
                'estimator__solver': ['liblinear', 'saga']
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'ANN': {
            'model': MLPClassifier(max_iter=5000),  # Increase max iterations
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
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
            'model': None,  # Custom implementation
            'param_grid': {}
        },
        'MCDA': {
            'model': None,  # Custom implementation
            'param_grid': {}
        },
        'AHP': {
            'model': None,  # Custom implementation
            'param_grid': {}
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
