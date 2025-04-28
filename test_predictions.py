import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load all trained models
models = {
    'SVM': joblib.load('output/models/svm.joblib'),
    'Random Forest': joblib.load('output/models/random_forest.joblib'),
    'KNN': joblib.load('output/models/knn.joblib'),
    'ANN': joblib.load('output/models/ann.joblib'),
    'XGBoost': joblib.load('output/models/xgboost.joblib'),
    'Gradient Boosting': joblib.load('output/models/gradient_boosting.joblib'),
    'Logistic Regression': joblib.load('output/models/logistic_regression.joblib'),  # Added Logistic Regression
    'AHP': joblib.load('output/models/ahp_weights.joblib'),
    'Fuzzy Technique': joblib.load('output/models/fuzzy_technique.joblib'),
    'MCDA': joblib.load('output/models/mcda_weights.joblib')
}

# Reverse mapping for flood status
flood_status_map_reverse = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High"
}

def print_menu():
    """Display the main menu"""
    print("\n" + "=" * 50)
    print("FLOOD PREDICTION TESTING")
    print("=" * 50)
    print("\nAvailable Options:")
    print("-----------------")
    print("1. Enter custom values")
    print("2. Test a specific record from dataset")
    print("3. Test a random record from dataset")
    print("4. Exit")
    print("\n" + "=" * 50)

def get_user_input():
    """Get input values from user"""
    print("\nPlease enter values for prediction:")
    print("-" * 30)
    features = {
        'Rainfall': {'min': 0, 'max': 100, 'unit': 'mm/hr'},
        'Water Level': {'min': 0, 'max': 5000, 'unit': 'mm'},
        'Elevation': {'min': 0, 'max': 1000, 'unit': 'm'},
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

def test_with_custom_input(input_values):
    """Test all models with custom input values"""
    # Convert input values to feature array in the correct order
    features = np.array([[input_values['Rainfall'], input_values['Water Level'], input_values['Elevation'], input_values['Slope'], input_values['Distance from River']]])
    
    print("\nInput Values:")
    print("-" * 50)
    for feature, value in input_values.items():
        print(f"{feature}: {value}")
    
    print("\nModel Predictions:")
    print("-" * 50)
    print(f"{'Model':<20} {'Prediction':<15} {'Risk Levels/Probabilities':<50}")
    print("-" * 85)
    
    for model_name, model in models.items():
        try:
            if model_name in ['AHP', 'Fuzzy Technique', 'MCDA']:
                if model_name == 'AHP':
                    weights = model  # Load the AHP weights
                    # Normalize the input features
                    scaler = MinMaxScaler()
                    features_norm = scaler.fit_transform(features)
                    score = np.dot(features_norm, weights)[0]
                    risk_level = flood_status_map_reverse[2] if score > np.percentile(weights, 66.67) else \
                                 flood_status_map_reverse[1] if score > np.percentile(weights, 33.33) else \
                                 flood_status_map_reverse[0]
                    print(f"{model_name:<20} {risk_level:<15} Score: {score:.2f}")
                
                elif model_name == 'Fuzzy Technique':
                    flood_ctrl = model  # Load the fuzzy control system
                    flood_sim = ctrl.ControlSystemSimulation(flood_ctrl)  # Create a simulation instance
                    flood_sim.input['rainfall'] = features[0][0]
                    flood_sim.input['water_level'] = features[0][1]
                    flood_sim.input['elevation'] = features[0][2]
                    flood_sim.input['slope'] = features[0][3]
                    flood_sim.input['distance'] = features[0][4]
                    flood_sim.compute()
                    risk_value = flood_sim.output['flood_risk']
                    risk_level = flood_status_map_reverse[int(round(risk_value))]
                    print(f"{model_name:<20} {risk_level:<15} Risk Value: {risk_value:.2f}")
                
                elif model_name == 'MCDA':
                    weights = model
                    scaler = MinMaxScaler()
                    features_norm = scaler.fit_transform(features)
                    score = np.dot(features_norm, weights)[0]
                    risk_level = flood_status_map_reverse[2] if score > 0.66 else flood_status_map_reverse[1] if score > 0.33 else flood_status_map_reverse[0]
                    print(f"{model_name:<20} {risk_level:<15} Score: {score:.2%}")
            else:
                # For ML models
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                risk_level = flood_status_map_reverse[prediction]
                probs_str = ", ".join([f"{flood_status_map_reverse[i]}: {prob:.1%}" for i, prob in enumerate(probabilities)])
                print(f"{model_name:<20} {risk_level:<15} {probs_str}")
        except Exception as e:
            print(f"{model_name:<20} Error: {str(e)}")

def test_from_dataset(record_index=None):
    """Test using a record from the dataset"""
    # Load the dataset
    df = pd.read_csv('data/THESIS - GIS DATA - FLOOD SCENARIOS_UPDATED.csv')
    
    # Map descriptive labels to numeric values
    flood_status_map = {
        "Very Low": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }
    df['Flood Status'] = df['Flood Status'].map(flood_status_map)
    
    if record_index is None:
        # Choose a random record
        record_index = np.random.randint(0, len(df))
    elif record_index >= len(df):
        print(f"Error: Record index {record_index} is out of range. Dataset has {len(df)} records.")
        return
    
    record = df.iloc[record_index]
    input_values = {
        'Rainfall': record['Rainfall'],
        'Water Level': record['Water Level'],
        'Elevation': record['Elevation'],
        'Slope': record['Slope'],
        'Distance from River': record['Distance from River']
    }
    
    print(f"\nTesting Record {record_index}:")
    print("-" * 30)
    print(f"Actual Class: {flood_status_map_reverse[record['Flood Status']]}")
    
    test_with_custom_input(input_values)

if __name__ == "__main__":
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            input_values = get_user_input()
            test_with_custom_input(input_values)
        elif choice == "2":
            try:
                record_index = int(input("Enter the record index to test: "))
                test_from_dataset(record_index)
            except ValueError:
                print("Please enter a valid number")
        elif choice == "3":
            test_from_dataset()
        elif choice == "4":
            print("\nThank you for using the Flood Prediction Testing tool!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")
        
        if choice in ["1", "2", "3"]:
            input("\nPress Enter to continue...")