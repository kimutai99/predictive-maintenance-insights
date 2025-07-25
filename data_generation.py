import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
num_machines = 200
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 7, 16)
date_range = (end_date - start_date).days

# Machine Types and Components
machine_types = ['Excavator', 'Haul Truck', 'Dozer', 'Grader', 'Loader']
components = {
    'Excavator': ['Hydraulic Pump', 'Engine', 'Undercarriage', 'Swing Motor'],
    'Haul Truck': ['Engine', 'Transmission', 'Tires', 'Brake System'],
    'Dozer': ['Blade Hydraulics', 'Engine', 'Tracks', 'Final Drives'],
    'Grader': ['Moldboard', 'Engine', 'Hydraulic System', 'Tires'],
    'Loader': ['Bucket', 'Engine', 'Hydraulic System', 'Transmission']
}

# Generate Base Data
data = {
    'Machine_ID': [f'M{1000 + i}' for i in range(num_machines)],
    'Machine_Type': np.random.choice(machine_types, num_machines),
    'Age_Years': np.random.randint(1, 15, num_machines),
    'Last_Service_Date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_machines)],
    'Repairs_Count': np.random.randint(0, 20, num_machines)
}
df = pd.DataFrame(data)

# Generate Time-Series Sensor Data
records = []
for i, row in df.iterrows():
    for day in range(date_range):
        current_date = start_date + timedelta(days=day)
        
        # Simulate sensor data with some noise and drift
        temp = 60 + (row['Age_Years'] * 1.5) + np.random.normal(0, 5) + (day / 100)
        pressure = 300 + (row['Repairs_Count'] * 2) + np.random.normal(0, 10) - (day / 150)
        vibration = 5 + (row['Age_Years'] * 0.2) + np.random.normal(0, 1) + (day / 200)
        
        # Environmental Factors
        humidity = 40 + np.random.normal(0, 5)
        ambient_temp = 25 + np.random.normal(0, 3)

        records.append({
            'Timestamp': current_date,
            'Machine_ID': row['Machine_ID'],
            'Machine_Type': row['Machine_Type'],
            'Age_Years': row['Age_Years'],
            'Last_Service_Date': row['Last_Service_Date'],
            'Repairs_Count': row['Repairs_Count'],
            'Temperature_C': temp,
            'Pressure_PSI': pressure,
            'Vibration_mm_s': vibration,
            'Humidity_Percent': humidity,
            'Ambient_Temp_C': ambient_temp,
            'Failure': 0,
            'Failure_Component': 'None'
        })

time_series_df = pd.DataFrame(records)

# Introduce Failures
num_failures = int(num_machines * (date_range / 365) * 0.5) # Average of 1 failure per machine per 2 years
failure_indices = np.random.choice(time_series_df.index, num_failures, replace=False)

for idx in failure_indices:
    machine_type = time_series_df.loc[idx, 'Machine_Type']
    component_to_fail = np.random.choice(components[machine_type])
    
    # Mark the failure and the component
    time_series_df.loc[idx, 'Failure'] = 1
    time_series_df.loc[idx, 'Failure_Component'] = component_to_fail

    # Make sensor data more extreme leading up to the failure
    for i in range(1, 31): # 30 days before failure
        if idx - i >= 0:
            time_series_df.loc[idx - i, 'Temperature_C'] *= (1 + (31 - i) * 0.005)
            time_series_df.loc[idx - i, 'Vibration_mm_s'] *= (1 + (31 - i) * 0.01)


# Define Target for Predictive Maintenance (predicting failure in the next 14 days)
time_series_df['Failure_in_14_Days'] = time_series_df.groupby('Machine_ID')['Failure'].transform(
    lambda x: x.rolling(window=14, min_periods=1).max().shift(-14)
).fillna(0)


# Save to CSV
output_path = r'C:\Users\HP\MachineFailurePredictor\mining_equipment_data.csv'
time_series_df.to_csv(output_path, index=False)

print(f"Dataset generated and saved to {output_path}")
