
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

# Configuration
num_machines = 50
data_points_per_machine = 500
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2025-01-01')

# Machine Types and Components
machine_types = ['Excavator', 'Bulldozer', 'Haul Truck', 'Grader', 'Loader']
components = {
    'Excavator': ['Hydraulic Pump', 'Engine', 'Undercarriage', 'Swing Motor'],
    'Bulldozer': ['Engine', 'Transmission', 'Blade Lift Cylinder', 'Final Drives'],
    'Haul Truck': ['Engine', 'Transmission', 'Tire', 'Suspension'],
    'Grader': ['Circle Drive Motor', 'Blade Side Shift Cylinder', 'Engine', 'Transmission'],
    'Loader': ['Bucket', 'Hydraulic Cylinder', 'Engine', 'Transmission']
}

# Generate Machine Details
def generate_machine_details(num_machines):
    machines = []
    for i in range(1, num_machines + 1):
        machine_type = random.choice(machine_types)
        machines.append({
            'machine_id': f'M{i:03d}',
            'machine_type': machine_type,
            'model': f'{machine_type}-{random.choice(["A", "B", "C"])}-{random.randint(100, 999)}',
            'age_years': random.randint(1, 15),
            'manufacturer': fake.company()
        })
    return pd.DataFrame(machines)

# Generate Synthetic Data
def generate_synthetic_data(machines_df):
    data = []
    for _, machine in machines_df.iterrows():
        machine_id = machine['machine_id']
        machine_type = machine['machine_type']
        age = machine['age_years']
        
        # Generate timestamps
        timestamps = pd.to_datetime(np.linspace(start_date.value, end_date.value, data_points_per_machine))
        
        # Base sensor values
        temp_base = 60 + age * 1.5
        vibration_base = 2 + age * 0.1
        pressure_base = 1000 + age * 10
        
        # Failure simulation
        failure_chance = 0.01 + (age / 15) * 0.05
        is_failure_imminent = False
        failure_countdown = 0
        
        for ts in timestamps:
            if not is_failure_imminent and random.random() < failure_chance:
                is_failure_imminent = True
                failure_countdown = random.randint(10, 30) # Days until failure
                failed_component = random.choice(components[machine_type])

            # Sensor data simulation
            temp = np.random.normal(temp_base, 5)
            vibration = np.random.normal(vibration_base, 0.5)
            pressure = np.random.normal(pressure_base, 50)
            
            if is_failure_imminent:
                increase_factor = 1 - (failure_countdown / 30)
                temp += increase_factor * 20
                vibration += increase_factor * 2
                pressure += increase_factor * 100
                failure_countdown -= 1
                
            failure_in_14_days = 1 if is_failure_imminent and failure_countdown <= 14 else 0
            
            data.append({
                'timestamp': ts,
                'machine_id': machine_id,
                'temperature_c': temp,
                'vibration_mm_s': vibration,
                'pressure_psi': pressure,
                'rotational_speed_rpm': np.random.normal(1500, 100),
                'load_weight_tonnes': np.random.normal(50, 10),
                'ambient_temp_c': np.random.normal(25, 5),
                'humidity_percent': np.random.normal(60, 10),
                'failure_in_14_days': failure_in_14_days,
                'failed_component': failed_component if is_failure_imminent and failure_countdown <= 0 else 'None'
            })
            
            if is_failure_imminent and failure_countdown <= 0:
                is_failure_imminent = False

    return pd.DataFrame(data)

if __name__ == '__main__':
    machines = generate_machine_details(num_machines)
    synthetic_data = generate_synthetic_data(machines)
    
    # Merge machine details with synthetic data
    final_df = pd.merge(synthetic_data, machines, on='machine_id')
    
    # Save to CSV
    final_df.to_csv('C:/Users/HP/MachineFailurePredictor/data/synthetic_mining_data.csv', index=False)
    print("Data generation complete. Saved to data/synthetic_mining_data.csv")
