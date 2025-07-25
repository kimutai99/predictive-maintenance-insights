
import pandas as pd
import numpy as np
import random
from faker import Faker

# This script generates different sample datasets for the Streamlit app.

fake = Faker()

# --- Configuration ---
num_machines = 50
data_points_per_machine = 100
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-06-01')
machine_types = ['Excavator', 'Bulldozer', 'Haul Truck', 'Grader', 'Loader']

# --- Base Generation Functions ---
def generate_machine_details(num_machines, seed):
    random.seed(seed)
    np.random.seed(seed)
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

def generate_synthetic_data(machines_df):
    data = []
    for _, machine in machines_df.iterrows():
        machine_id = machine['machine_id']
        age = machine['age_years']
        timestamps = pd.to_datetime(np.linspace(start_date.value, end_date.value, data_points_per_machine))
        
        temp_base = 60 + age * 1.2
        vibration_base = 2 + age * 0.15
        failure_chance = 0.05 + (age / 15) * 0.1

        for ts in timestamps:
            temp = np.random.normal(temp_base, 5)
            vibration = np.random.normal(vibration_base, 0.5)

            # Simulate random spikes that could indicate issues
            if random.random() < failure_chance:
                temp += random.uniform(5, 20)
                vibration += random.uniform(1, 3)

            data.append({
                'timestamp': ts, 'machine_id': machine_id,
                'temperature_c': temp, 'vibration_mm_s': vibration,
                'pressure_psi': np.random.normal(1000, 50), 'rotational_speed_rpm': np.random.normal(1500, 100),
                'load_weight_tonnes': np.random.normal(50, 10), 'ambient_temp_c': np.random.normal(25, 5),
                'humidity_percent': np.random.normal(60, 10)
            })

    return pd.DataFrame(data)

if __name__ == '__main__':
    print("Generating sample datasets...")
    # Generate three similar but distinct datasets using different random seeds
    for i in range(1, 4):
        seed = 42 * i
        machines = generate_machine_details(num_machines, seed)
        sample_data = generate_synthetic_data(machines)
        final_df = pd.merge(sample_data, machines, on='machine_id')
        # The 'failed_component' and 'failure_in_14_days' columns are not included
        final_df.to_csv(f'C:/Users/HP/MachineFailurePredictor/data/sample_data_{i}.csv', index=False)
        print(f"-> Created sample_data_{i}.csv")

    print("Sample dataset generation complete.")
