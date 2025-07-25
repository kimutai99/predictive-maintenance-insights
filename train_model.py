import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from generate_data import create_dataset # Import the data generation function

def feature_engineering(df):
    """Creates features for the model."""
    # Rolling averages for sensor data
    for col in ['temperature_c', 'vibration_mm_s', 'pressure_psi', 'rotational_speed_rpm', 'load_weight_tonnes']:
        df[f'{col}_rolling_avg_7d'] = df.groupby('machine_id')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Time-based features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    return df

def train_and_evaluate_models(df):
    """Trains and evaluates the failure and component prediction models."""
    # Encode categorical features
    categorical_cols = ['machine_type', 'model', 'manufacturer']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Define features (X) and targets (y)
    features = ['age_years', 'day_of_week', 'month'] + \
               [col for col in df.columns if '_rolling_avg_7d' in col] + categorical_cols
    
    # Add sensor data directly as features
    features.extend(['temperature_c', 'vibration_mm_s', 'pressure_psi', 'rotational_speed_rpm', 'load_weight_tonnes'])

    # Drop rows where features might be NaN (e.g., from rolling averages)
    df.dropna(subset=features, inplace=True)

    X = df[features]
    y_failure = df['failure_in_14_days']
    y_component = df['failing_component']

    # Split data
    X_train, X_test, y_failure_train, y_failure_test, y_component_train, y_component_test = train_test_split(
        X, y_failure, y_component, test_size=0.2, random_state=42, stratify=y_failure
    )


    # Train failure prediction model
    print("Training failure prediction model...")
    failure_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=20)
    failure_model.fit(X_train, y_failure_train)

    # Train component prediction model
    print("Training component prediction model...")
    # We only train on data where a failure is predicted
    X_failure_cases = X_train[y_failure_train == 1]
    y_component_failure_cases = y_component_train[y_failure_train == 1].dropna()
    X_failure_cases = X_failure_cases.loc[y_component_failure_cases.index]

    le_component = LabelEncoder()
    y_component_encoded = le_component.fit_transform(y_component_failure_cases)

    component_model = RandomForestClassifier(n_estimators=100, random_state=42)
    component_model.fit(X_failure_cases, y_component_encoded)

    # Evaluate models
    print("\n--- Failure Prediction Model Evaluation ---")
    y_failure_pred = failure_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_failure_test, y_failure_pred):.4f}")
    print(classification_report(y_failure_test, y_failure_pred))

    print("\n--- Component Prediction Model Evaluation ---")
    X_test_failures = X_test[y_failure_test == 1]
    y_component_test_failures = y_component_test[y_failure_test == 1].dropna()
    X_test_failures = X_test_failures.loc[y_component_test_failures.index]

    if not X_test_failures.empty:
        y_component_pred_encoded = component_model.predict(X_test_failures)
        y_component_pred = le_component.inverse_transform(y_component_pred_encoded)
        
        print(f"Accuracy: {accuracy_score(y_component_test_failures, y_component_pred):.4f}")
        print(classification_report(y_component_test_failures, y_component_pred))

    return failure_model, component_model, le_component, X.columns, encoders

def main():
    """Main function to generate data and run the training pipeline."""
    # Generate a large, comprehensive dataset for training
    training_data_filename = "training_data.csv"
    create_dataset(training_data_filename, num_machines=100, n_days=365, num_manufacturers=10, high_risk=True)
    
    training_data_path = os.path.join('data', training_data_filename)
    df = pd.read_csv(training_data_path, parse_dates=['timestamp'])

    df = feature_engineering(df)
    
    failure_model, component_model, le_component, features, encoders = train_and_evaluate_models(df)
    
    # Save models to the `models` directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump({'model': failure_model, 'encoders': encoders, 'features': features}, os.path.join(models_dir, 'failure_prediction_model.joblib'))
    joblib.dump({'model': component_model, 'label_encoder': le_component, 'features': features}, os.path.join(models_dir, 'component_prediction_model.joblib'))
    print("Models saved successfully.")

if __name__ == "__main__":
    main()
