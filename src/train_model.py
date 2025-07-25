
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/HP/MachineFailurePredictor/data/machine_failure_data.csv')

# Preprocessing
# Convert timestamp to datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# Filter for failures
df_failures = df[df['Failure_in_14_Days'] == 1].copy()

# Label encode the target variable
le = LabelEncoder()
df_failures['Failed_Component_Encoded'] = le.fit_transform(df_failures['Failed_Component'])

# Select features and target
features = ['Age_Days', 'Temperature_C', 'Humidity_pct', 'Vibration_mm_s', 'Pressure_psi', 'Maintenance_Events', 'Days_Since_Maintenance']
target = 'Failed_Component_Encoded'

X = df_failures[features]
y = df_failures[target]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and the label encoder
joblib.dump(model, 'C:/Users/HP/MachineFailurePredictor/src/models/machine_failure_model.pkl')
joblib.dump(le, 'C:/Users/HP/MachineFailurePredictor/src/models/label_encoder.pkl')

print("Model trained and saved successfully!")
