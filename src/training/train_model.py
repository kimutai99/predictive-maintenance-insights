import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_models():
    # Load data
    df = pd.read_csv('C:/Users/HP/MachineFailurePredictor/data/synthetic_mining_data.csv', parse_dates=['timestamp'])

    # Feature Engineering
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Label Encoding
    encoders = {}
    for col in ['machine_type', 'model', 'manufacturer']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # --- Failure Prediction Model ---
    features = [
        'age_years', 'temperature_c', 'vibration_mm_s', 'pressure_psi',
        'rotational_speed_rpm', 'load_weight_tonnes', 'ambient_temp_c',
        'humidity_percent', 'day_of_week', 'month',
        'machine_type', 'model', 'manufacturer'
    ]
    target_failure = 'failure_in_14_days'

    X = df[features]
    y = df[target_failure]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    failure_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    failure_model.fit(X_train, y_train)

    y_pred = failure_model.predict(X_test)
    print("--- Failure Prediction Model ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save failure model and encoders
    joblib.dump({'model': failure_model, 'encoders': encoders, 'features': features}, 'C:/Users/HP/MachineFailurePredictor/models/failure_prediction_model.joblib')

    # --- Component Prediction Model ---
    df_failures = df[df['failed_component'] != 'None']
    target_component = 'failed_component'

    le_component = LabelEncoder()
    df_failures[target_component] = le_component.fit_transform(df_failures[target_component])

    X_comp = df_failures[features]
    y_comp = df_failures[target_component]

    if len(df_failures) > 10:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42, stratify=y_comp)

        component_model = RandomForestClassifier(n_estimators=100, random_state=42)
        component_model.fit(X_train_c, y_train_c)

        y_pred_c = component_model.predict(X_test_c)
        print("\n--- Component Prediction Model ---")
        print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c)}")
        print(classification_report(y_test_c, y_pred_c))

        # Save component model and encoder
        joblib.dump({'model': component_model, 'label_encoder': le_component, 'features': features}, 'C:/Users/HP/MachineFailurePredictor/models/component_prediction_model.joblib')

if __name__ == '__main__':
    train_models()
