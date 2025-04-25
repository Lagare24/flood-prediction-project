import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from skfuzzy import control as ctrl
import skfuzzy as fuzz  # Add this import for fuzzy functions
import warnings
import joblib
import os

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

    # Convert date to datetime with dayfirst=True for DD/MM/YYYY format
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)

    # Create flood label based on water level threshold
    # Adjusted threshold to make the problem more challenging
    # Using 75th percentile of water level as threshold
    water_level_threshold = df['Water level'].quantile(0.75)
    print(f"\nUsing water level threshold: {water_level_threshold:.2f}")
    df['Flood'] = (df['Water level'] > water_level_threshold).astype(int)

    # Extract features
    X = df[['Rainfall Rate', 'Water level', 'elevation', 'Slope', 'Distance from River']]
    y = df['Flood']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df


# Function to perform hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Function to implement Fuzzy Technique
def implement_fuzzy_system(X_train, y_train, X_test):
    # Define fuzzy variables
    rainfall = ctrl.Antecedent(np.arange(0, 100, 1), 'rainfall')
    water_level = ctrl.Antecedent(np.arange(0, 2000, 1), 'water_level')
    flood_risk = ctrl.Consequent(np.arange(0, 1, 0.1), 'flood_risk')

    # Define membership functions
    rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 50])
    rainfall['medium'] = fuzz.trimf(rainfall.universe, [0, 50, 100])
    rainfall['high'] = fuzz.trimf(rainfall.universe, [50, 100, 100])

    water_level['low'] = fuzz.trimf(water_level.universe, [0, 0, 1000])
    water_level['medium'] = fuzz.trimf(water_level.universe, [0, 1000, 2000])
    water_level['high'] = fuzz.trimf(water_level.universe, [1000, 2000, 2000])

    flood_risk['low'] = fuzz.trimf(flood_risk.universe, [0, 0, 0.5])
    flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [0, 0.5, 1])
    flood_risk['high'] = fuzz.trimf(flood_risk.universe, [0.5, 1, 1])

    # Define rules
    rule1 = ctrl.Rule(rainfall['low'] & water_level['low'], flood_risk['low'])
    rule2 = ctrl.Rule(rainfall['medium'] & water_level['medium'], flood_risk['medium'])
    rule3 = ctrl.Rule(rainfall['high'] & water_level['high'], flood_risk['high'])

    # Create control system
    flood_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)

    # Make predictions
    y_pred = []
    for i in range(len(X_test)):
        flood_sim.input['rainfall'] = X_test[i, 0]
        flood_sim.input['water_level'] = X_test[i, 1]
        flood_sim.compute()
        y_pred.append(flood_sim.output['flood_risk'])

    return np.array(y_pred)


# Function to calculate entropy weights
def calculate_entropy_weights(X):
    # Normalize the data
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    # Calculate entropy for each criterion
    n_samples, n_criteria = X_norm.shape
    entropies = []
    for j in range(n_criteria):
        # Add small constant to avoid log(0)
        pij = X_norm[:, j] + 1e-10
        pij = pij / pij.sum()
        ej = -np.sum(pij * np.log(pij)) / np.log(n_samples)
        entropies.append(ej)
    
    # Calculate criteria weights
    entropies = np.array(entropies)
    weights = (1 - entropies) / ((1 - entropies).sum())
    return weights


# Function to implement MCDA
def implement_mcda(X_train, y_train, X_test):
    # Calculate weights using entropy method
    weights = calculate_entropy_weights(X_train)
    
    # Calculate weighted sum for test data
    scores = np.dot(X_test, weights)
    
    return scores, weights  # Return both scores and weights


# Function to implement AHP
def implement_ahp(X_train, y_train, X_test):
    # Simple weighted sum approach for AHP
    weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # Example weights
    scores = np.dot(X_test, weights)
    return scores


# Function to save model and predictions
def save_model_and_predictions(model, model_name, X_test, y_test, y_pred, y_pred_proba):
    # Save the model
    if model is not None:
        joblib.dump(model, f'output/models/{model_name.lower().replace(" ", "_")}.joblib')
    elif model_name == 'Fuzzy Technique':
        # Save fuzzy system parameters
        fuzzy_params = {
            'rainfall_range': (0, 100),
            'water_level_range': (0, 2000),
            'flood_risk_range': (0, 1),
            'rules': [
                "IF rainfall IS low AND water_level IS low THEN flood_risk IS low",
                "IF rainfall IS medium AND water_level IS medium THEN flood_risk IS medium",
                "IF rainfall IS high AND water_level IS high THEN flood_risk IS high"
            ]
        }
        joblib.dump(fuzzy_params, f'output/models/fuzzy_technique.joblib')
    elif model_name == 'MCDA':
        # Save MCDA parameters
        _, weights = implement_mcda(X_test, y_test, X_test)  # Get weights using test data
        mcda_params = {
            'method': 'entropy_weighting',
            'normalization': 'minmax',
            'criteria_weights': weights
        }
        joblib.dump(mcda_params, f'output/models/mcda.joblib')
    elif model_name == 'AHP':
        # Save AHP parameters
        ahp_params = {
            'method': 'weighted_sum',
            'weights': np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # Example weights
        }
        joblib.dump(ahp_params, f'output/models/ahp.joblib')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_pred_proba
    })
    predictions_df.to_csv(f'output/predictions/{model_name.lower().replace(" ", "_")}_predictions.csv', index=False)


# Function to train and evaluate model
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    if model_name == 'Fuzzy Technique':
        y_pred = implement_fuzzy_system(X_train, y_train, X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_pred_proba = y_pred
    elif model_name == 'MCDA':
        y_pred, _ = implement_mcda(X_train, y_train, X_test)
        y_pred_binary = (y_pred > np.mean(y_pred)).astype(int)
        y_pred_proba = y_pred
    elif model_name == 'AHP':
        y_pred = implement_ahp(X_train, y_train, X_test)
        y_pred_binary = (y_pred > np.mean(y_pred)).astype(int)
        y_pred_proba = y_pred
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_binary = y_pred
    
    # Save model and predictions
    save_model_and_predictions(model, model_name, X_test, y_test, y_pred_binary, y_pred_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'model': model
    }


# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
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
    # Load data
    csv_path = "data/THESIS - GIS DATA - ALL DATA.csv"
    df = load_data(csv_path)

    # Preprocess data
    X, y, df = preprocess_data(df)
    feature_names = ['Rainfall Rate', 'Water level', 'elevation', 'Slope', 'Distance from River']

    # Print data summary
    print("\nData Summary:")
    print("-" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Flood events: {y.sum()} ({y.sum() / len(y) * 100:.2f}%)")
    print(f"Non-flood events: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their hyperparameter grids
    models_config = {
        'SVM': {
            'model': SVC(probability=True),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
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
            'model': MLPClassifier(),
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'max_iter': [1000, 2000]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
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
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 100)
    for result in results:
        auc_value = result['roc_auc'] if result['roc_auc'] is not None else 'N/A'
        print(f"{result['model_name']:<20} {result['accuracy']:.4f} {result['precision']:.4f} "
              f"{result['recall']:.4f} {result['f1']:.4f} {auc_value}")


if __name__ == "__main__":
    main()
