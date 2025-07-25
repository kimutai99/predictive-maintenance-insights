import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data_path = "C:/Users/HP/MachineFailurePredictor/mining_equipment_data.csv"
df = pd.read_csv(data_path)

# Feature Engineering
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])

# Time-based features
df['Days_Since_Service'] = (df['Timestamp'] - df['Last_Service_Date']).dt.days
df['Timestamp_day'] = df['Timestamp'].dt.day
df['Timestamp_month'] = df['Timestamp'].dt.month
df['Timestamp_year'] = df['Timestamp'].dt.year

# Label Encoding for categorical features
le_machine_type = LabelEncoder()
df['Machine_Type_Encoded'] = le_machine_type.fit_transform(df['Machine_Type'])

le_component = LabelEncoder()
df['Failure_Component_Encoded'] = le_component.fit_transform(df['Failure_Component'])

# Define features (X) and target (y)
features = [
    'Age_Years', 'Repairs_Count', 'Temperature_C', 'Pressure_PSI', 
    'Vibration_mm_s', 'Humidity_Percent', 'Ambient_Temp_C',
    'Days_Since_Service', 'Timestamp_day', 'Timestamp_month', 'Timestamp_year',
    'Machine_Type_Encoded'
]

target_failure = 'Failure_in_14_Days'
target_component = 'Failure_Component_Encoded'

# Drop rows with NaN in target (created by shift in data generation)
df.dropna(subset=[target_failure], inplace=True)


# --- Model 1: Predicting Failure in 14 Days ---
X = df[features]
y_failure = df[target_failure]

X_train, X_test, y_train, y_test = train_test_split(X, y_failure, test_size=0.2, random_state=42, stratify=y_failure)

failure_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
failure_model.fit(X_train, y_train)

y_pred_failure = failure_model.predict(X_test)
print("--- Failure Prediction Model ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_failure):.4f}")
print(classification_report(y_test, y_pred_failure))

# Save the failure prediction model
joblib.dump(failure_model, "C:/Users/HP/MachineFailurePredictor/failure_model.pkl")
joblib.dump(le_machine_type, "C:/Users/HP/MachineFailurePredictor/le_machine_type.pkl")


# --- Model 2: Predicting Component Failure ---
# We only train on data where a failure occurred to predict the component
df_failures = df[df['Failure'] == 1]

X_component = df_failures[features]
y_component = df_failures[target_component]

if len(df_failures) > 10: # Ensure there is enough data to train
    X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
        X_component, y_component, test_size=0.2, random_state=42, stratify=y_component
    )

    component_model = RandomForestClassifier(n_estimators=100, random_state=42)
    component_model.fit(X_train_comp, y_train_comp)

    y_pred_component = component_model.predict(X_test_comp)
    print("\n--- Component Prediction Model ---")
    print(f"Accuracy: {accuracy_score(y_test_comp, y_pred_component):.4f}")
    print(classification_report(y_test_comp, y_pred_component, zero_division=0))
    
    # Save the component prediction model and encoder
    joblib.dump(component_model, "C:/Users/HP/MachineFailurePredictor/component_model.pkl")
    joblib.dump(le_component, "C:/Users/HP/MachineFailurePredictor/le_component.pkl")

print("\nModels trained and saved successfully!")
