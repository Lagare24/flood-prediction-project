import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_user_input():
    """Get input values from user"""
    print("\nPlease enter values for prediction:")
    features = {
        'Rainfall Rate': {'min': 0, 'max': 100, 'unit': 'mm/hr'},
        'Water level': {'min': 0, 'max': 5000, 'unit': 'mm'},
        'elevation': {'min': 0, 'max': 1000, 'unit': 'm'},
        'Slope': {'min': 0, 'max': 90, 'unit': 'degrees'},
        'Distance from River': {'min': 0, 'max': 5000, 'unit': 'm'}
    }
    
    input_values = {}
    for feature, limits in features.items():
        while True:
            try:
                value = float(input(f"Enter {feature} ({limits['unit']}, range {limits['min']}-{limits['max']}): "))
                if limits['min'] <= value <= limits['max']:
                    input_values[feature] = value
                    break
                else:
                    print(f"Value must be between {limits['min']} and {limits['max']}")
            except ValueError:
                print("Please enter a valid number")
    
    return input_values

def load_model(model_name):
    """Load a saved model from the output/models directory"""
    model_path = f'output/models/{model_name.lower().replace(" ", "_")}.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def fuzzy_prediction(input_values):
    """Implement fuzzy logic prediction"""
    # Create fuzzy variables
    rainfall = ctrl.Antecedent(np.arange(0, 100, 1), 'rainfall')
    water_level = ctrl.Antecedent(np.arange(0, 5000, 1), 'water_level')
    flood_risk = ctrl.Consequent(np.arange(0, 1, 0.01), 'flood_risk')
    
    # Define membership functions
    rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 50])
    rainfall['medium'] = fuzz.trimf(rainfall.universe, [0, 50, 100])
    rainfall['high'] = fuzz.trimf(rainfall.universe, [50, 100, 100])
    
    water_level['low'] = fuzz.trimf(water_level.universe, [0, 0, 2500])
    water_level['medium'] = fuzz.trimf(water_level.universe, [0, 2500, 5000])
    water_level['high'] = fuzz.trimf(water_level.universe, [2500, 5000, 5000])
    
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
    
    # Input values
    flood_sim.input['rainfall'] = input_values['Rainfall Rate']
    flood_sim.input['water_level'] = input_values['Water level']
    
    # Compute result
    try:
        flood_sim.compute()
        risk = flood_sim.output['flood_risk']
        return risk >= 0.5, risk
    except:
        return None, None

def mcda_prediction(input_values):
    """Implement MCDA prediction using entropy weighting"""
    # Convert input to array
    X = np.array([[
        input_values['Rainfall Rate'],
        input_values['Water level'],
        input_values['elevation'],
        input_values['Slope'],
        input_values['Distance from River']
    ]])
    
    # Load saved weights
    mcda_params = load_model('MCDA')
    weights = mcda_params['criteria_weights']
    
    # Normalize input
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    # Calculate weighted sum
    score = np.dot(X_norm, weights)[0]
    
    # Predict flood if score is above mean (0.5)
    return score > 0.5, score

def ahp_prediction(input_values):
    """Implement AHP prediction"""
    # Load saved weights
    ahp_params = load_model('AHP')
    weights = ahp_params['weights']
    
    # Convert input to array
    X = np.array([[
        input_values['Rainfall Rate'],
        input_values['Water level'],
        input_values['elevation'],
        input_values['Slope'],
        input_values['Distance from River']
    ]])
    
    # Normalize input
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    # Calculate weighted sum
    score = np.dot(X_norm, weights)[0]
    
    # Predict flood if score is above mean (0.5)
    return score > 0.5, score

def test_model(model_name, input_values):
    """Test a specific model with custom input values"""
    # Define feature names in order
    features = ['Rainfall Rate', 'Water level', 'elevation', 'Slope', 'Distance from River']
    
    # Create input array in correct order
    X = np.array([[input_values[feature] for feature in features]])
    
    # Load and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Print input values
    print(f"\nModel: {model_name}")
    print("Input Values:")
    for feature, value in input_values.items():
        print(f"{feature}: {value}")
    
    try:
        if model_name == 'Fuzzy Technique':
            prediction, probability = fuzzy_prediction(input_values)
            if prediction is not None:
                print(f"\nPrediction: {'High Risk' if probability > 0.66 else 'Medium Risk' if probability > 0.33 else 'Low Risk'}")
                print(f"Flood Risk Score: {probability:.2%}")
            else:
                print("\nError: Could not compute fuzzy prediction")
        
        elif model_name == 'MCDA':
            prediction, score = mcda_prediction(input_values)
            print(f"\nPrediction: {'High Risk' if score > 0.66 else 'Medium Risk' if score > 0.33 else 'Low Risk'}")
            print(f"MCDA Score: {score:.2%}")
        
        elif model_name == 'AHP':
            prediction, score = ahp_prediction(input_values)
            print(f"\nPrediction: {'High Risk' if score > 0.66 else 'Medium Risk' if score > 0.33 else 'Low Risk'}")
            print(f"AHP Score: {score:.2%}")
        
        else:
            # For scikit-learn models
            model = load_model(model_name)
            prediction = model.predict(X_scaled)
            try:
                probability = model.predict_proba(X_scaled)[0]
                risk_level = 'High Risk' if probability[2] > 0.5 else 'Medium Risk' if probability[1] > 0.5 else 'Low Risk'
                print(f"\nPrediction: {risk_level}")
                print(f"Probabilities: Low: {probability[0]:.2%}, Medium: {probability[1]:.2%}, High: {probability[2]:.2%}")
            except:
                risk_level = 'High Risk' if prediction[0] == 2 else 'Medium Risk' if prediction[0] == 1 else 'Low Risk'
                print(f"\nPrediction: {risk_level}")
                print("Note: Detailed probabilities not available for this model")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("-" * 50)

def main():
    # List of all models with their descriptions
    models = [
        'Logistic Regression',
        'SVM',
        'Random Forest',
        'KNN',
        'ANN',
        'XGBoost',
        'Gradient Boosting',
        'Fuzzy Technique',
        'MCDA',
        'AHP'
    ]
    
    while True:
        print("\nFlood Prediction Model Testing")
        print("1. Use example values")
        print("2. Enter custom values")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            # Example input values
            test_input = {
                'Rainfall Rate': 50.0,
                'Water level': 2000.0,
                'elevation': 100.0,
                'Slope': 5.0,
                'Distance from River': 500.0
            }
        elif choice == '2':
            test_input = get_user_input()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        # Test each model
        for model in models:
            test_model(model, test_input)
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 