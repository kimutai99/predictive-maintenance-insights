
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
num_machines = 50
data_points_per_machine = 200
start_date = datetime(2023, 1, 1)

# Component failure probabilities (higher for components prone to failure)
component_failure_probs = {
    'Engine': 0.02,
    'Hydraulics': 0.03,
    'Electrical': 0.015,
    'Drivetrain': 0.025,
    'Cooling System': 0.01
}

# Generate data
data = []
for machine_id in range(1, num_machines + 1):
    maintenance_events = np.random.randint(0, 5)
    last_maintenance_date = start_date + timedelta(days=np.random.randint(0, 90))
    
    for i in range(data_points_per_machine):
        timestamp = start_date + timedelta(days=i)
        
        # Simulate sensor data with some noise and trends
        temperature = 80 + np.random.normal(0, 5) + (i / 50)
        humidity = 40 + np.random.normal(0, 10) + (i / 100)
        vibration = 0.5 + np.random.normal(0, 0.1) + (i / 200)
        pressure = 100 + np.random.normal(0, 2)
        
        # Age of the machine
        age_days = (timestamp - start_date).days
        
        # Maintenance history
        days_since_maintenance = (timestamp - last_maintenance_date).days
        
        # Failure prediction (within the next 14 days)
        failure_in_14_days = 0
        failed_component = 'None'
        
        # Check for component failure
        for component, prob in component_failure_probs.items():
            if np.random.rand() < prob and days_since_maintenance > 30:
                failure_in_14_days = 1
                failed_component = component
                # Reset maintenance date after a failure
                last_maintenance_date = timestamp
                break
        
        data.append([
            machine_id, 
            timestamp,
            age_days,
            temperature, 
            humidity,
            vibration,
            pressure,
            maintenance_events,
            days_since_maintenance,
            failure_in_14_days,
            failed_component
        ])

# Create DataFrame
columns = [
    'Machine_ID', 
    'Timestamp',
    'Age_Days',
    'Temperature_C', 
    'Humidity_pct',
    'Vibration_mm_s',
    'Pressure_psi',
    'Maintenance_Events',
    'Days_Since_Maintenance',
    'Failure_in_14_Days',
    'Failed_Component'
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('C:/Users/HP/MachineFailurePredictor/data/machine_failure_data.csv', index=False)

print("Dataset generated successfully!")
