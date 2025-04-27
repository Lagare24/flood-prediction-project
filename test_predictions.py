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
    'AHP': joblib.load('output/models/ahp.joblib'),
    'Fuzzy Technique': joblib.load('output/models/fuzzy_technique.joblib'),
    'MCDA': joblib.load('output/models/mcda.joblib')
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

def test_with_custom_input(input_values):
    """Test all models with custom input values"""
    # Convert input values to feature array in the correct order
    features = np.array([[
        input_values['Rainfall Rate'],
        input_values['Water level'],
        input_values['elevation'],
        input_values['Slope'],
        input_values['Distance from River']
    ]])
    
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
                    weights = model['weights']
                    score = np.dot(features, weights)[0]
                    risk_level = 'High Risk' if score > 0.66 else 'Medium Risk' if score > 0.33 else 'Low Risk'
                    print(f"{model_name:<20} {risk_level:<15} Score: {score:.2%}")
                
                elif model_name == 'Fuzzy Technique':
                    rainfall = features[0][0]
                    water_level = features[0][1]
                    score = (rainfall/100 + water_level/5000) / 2
                    risk_level = 'High Risk' if score > 0.66 else 'Medium Risk' if score > 0.33 else 'Low Risk'
                    print(f"{model_name:<20} {risk_level:<15} Score: {score:.2%}")
                
                elif model_name == 'MCDA':
                    scaler = MinMaxScaler()
                    features_norm = scaler.fit_transform(features)
                    weights = model['criteria_weights']
                    score = np.dot(features_norm, weights)[0]
                    risk_level = 'High Risk' if score > 0.66 else 'Medium Risk' if score > 0.33 else 'Low Risk'
                    print(f"{model_name:<20} {risk_level:<15} Score: {score:.2%}")
            else:
                # For ML models
                prediction = model.predict(features)[0]
                try:
                    probabilities = model.predict_proba(features)[0]
                    risk_level = 'High Risk' if prediction == 2 else 'Medium Risk' if prediction == 1 else 'Low Risk'
                    probs_str = f"Low: {probabilities[0]:.1%}, Med: {probabilities[1]:.1%}, High: {probabilities[2]:.1%}"
                    print(f"{model_name:<20} {risk_level:<15} {probs_str}")
                except:
                    risk_level = 'High Risk' if prediction == 2 else 'Medium Risk' if prediction == 1 else 'Low Risk'
                    print(f"{model_name:<20} {risk_level:<15} Probabilities not available")
        except Exception as e:
            print(f"{model_name:<20} Error: {str(e)}")

def test_from_dataset(record_index=None):
    """Test using a record from the dataset"""
    # Load the dataset
    df = pd.read_csv('data/THESIS - GIS DATA - ALL DATA.csv')
    
    if record_index is None:
        # Choose a random record
        record_index = np.random.randint(0, len(df))
    elif record_index >= len(df):
        print(f"Error: Record index {record_index} is out of range. Dataset has {len(df)} records.")
        return
    
    record = df.iloc[record_index]
    input_values = {
        'Rainfall Rate': record['Rainfall Rate'],
        'Water level': record['Water level'],
        'elevation': record['elevation'],
        'Slope': record['Slope'],
        'Distance from River': record['Distance from River']
    }
    
    print(f"\nTesting Record {record_index}:")
    print("-" * 30)
    print(f"Date: {record['Date']}")
    print(f"Actual Class: {record['Class']}")
    
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