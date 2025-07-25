import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Equipment details
MACHINE_TYPES = {
    'Haul Truck': ['CAT 797F', 'Komatsu 930E', 'Belaz 75710'],
    'Excavator': ['CAT 6090 FS', 'Liebherr R 9800', 'Hitachi EX8000'],
    'Dozer': ['CAT D11T', 'Komatsu D475A', 'Dressta TD-40']
}
MANUFACTURERS = [
    "Caterpillar", "Komatsu", "Liebherr", "Hitachi", "Belaz", 
    "Dressta", "John Deere", "Volvo CE", "Sandvik", "Metso"
]
COMPONENTS = {
    'Haul Truck': ['Engine', 'Transmission', 'Hydraulic System', 'Tires'],
    'Excavator': ['Engine', 'Hydraulic System', 'Undercarriage', 'Boom'],
    'Dozer': ['Engine', 'Transmission', 'Undercarriage', 'Blade']
}

# Sensor data simulation parameters (with more extreme failure values)
SENSOR_PARAMS = {
    'temperature_c': {'normal_mean': 85, 'normal_std': 5, 'fail_increase': 50},
    'vibration_mm_s': {'normal_mean': 0.6, 'normal_std': 0.15, 'fail_increase': 3.5},
    'pressure_psi': {'normal_mean': 2100, 'normal_std': 200, 'fail_increase': 900},
    'rotational_speed_rpm': {'normal_mean': 1600, 'normal_std': 150, 'fail_increase': -600},
    'load_weight_tonnes': {'normal_mean': 220, 'normal_std': 60},
    'ambient_temp_c': {'normal_mean': 25, 'normal_std': 8},
    'humidity_percent': {'normal_mean': 60, 'normal_std': 15}
}

def generate_base_data(num_machines, num_manufacturers):
    """Generates the base list of machines."""
    machines = []
    selected_manufacturers = np.random.choice(MANUFACTURERS, num_manufacturers, replace=False)
    for i in range(num_machines):
        machine_type = np.random.choice(list(MACHINE_TYPES.keys()))
        model = np.random.choice(MACHINE_TYPES[machine_type])
        age = np.random.randint(5, 20)
        manufacturer = np.random.choice(selected_manufacturers)
        machines.append({
            'machine_id': f'M{2001 + i}',
            'machine_type': machine_type,
            'model': model,
            'age_years': age,
            'manufacturer': manufacturer
        })
    return pd.DataFrame(machines)

def generate_time_series_data(machines_df, n_days, portion_critical=0.2):
    """Generates time-series sensor data with a portion of machines set to a critical failure path."""
    all_data = []
    data_start_date = datetime(2024, 1, 1)

    critical_machines_count = int(len(machines_df) * portion_critical)
    critical_machine_ids = machines_df.sample(n=critical_machines_count, random_state=42)['machine_id'].tolist()

    for _, machine in machines_df.iterrows():
        machine_id = machine['machine_id']
        machine_type = machine['machine_type']
        is_critical = machine_id in critical_machine_ids

        if is_critical:
            # For critical machines, schedule failure to be imminent at the end of the dataset
            days_to_failure = np.random.randint(n_days - 14, n_days)
            failing_component = np.random.choice(COMPONENTS[machine_type])
        else:
            # For other machines, failure can happen at any time, or not at all
            days_to_failure = np.random.exponential(scale=n_days * 2) + 15
            failing_component = None

        failure_date = data_start_date + timedelta(days=int(days_to_failure))

        current_date = data_start_date
        while current_date < (data_start_date + timedelta(days=n_days)):
            days_until_failure = (failure_date - current_date).days
            failure_in_14_days = 1 if (0 <= days_until_failure < 14) else 0

            row = {
                'timestamp': current_date,
                'machine_id': machine_id,
                'failure_in_14_days': failure_in_14_days,
                'failing_component': failing_component if failure_in_14_days else None
            }

            for sensor, params in SENSOR_PARAMS.items():
                mean = params['normal_mean']
                std = params['normal_std']
                if failure_in_14_days and 'fail_increase' in params:
                    severity = (14 - days_until_failure) / 14.0
                    mean += params['fail_increase'] * severity * 3.0
                row[sensor] = np.random.normal(mean, std)

            all_data.append(row)
            current_date += timedelta(days=1)

    return pd.DataFrame(all_data)

def create_dataset(file_path, num_machines, n_days, num_manufacturers, high_risk):
    """Main function to generate and save a single dataset."""
    print(f"Generating dataset for {file_path}...")
    machines_df = generate_base_data(num_machines, num_manufacturers)
    time_series_df = generate_time_series_data(machines_df, n_days, portion_critical=0.3 if high_risk else 0)
    
    final_df = pd.merge(time_series_df, machines_df, on='machine_id')
    
    cols = ['timestamp', 'machine_id', 'temperature_c', 'vibration_mm_s', 'pressure_psi',
            'rotational_speed_rpm', 'load_weight_tonnes', 'ambient_temp_c', 'humidity_percent',
            'machine_type', 'model', 'age_years', 'manufacturer',
            'failure_in_14_days', 'failing_component']
    final_df = final_df[cols]
    
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_df.to_csv(os.path.join(output_dir, file_path), index=False)
    print(f"Successfully saved {file_path} ({len(final_df)} records, {num_machines} machines)")
''

if __name__ == "__main__":
    # Dataset 1: 5000 records, 50 machines, 4 manufacturers, high risk
    create_dataset("sample_data_1.csv", num_machines=50, n_days=100, num_manufacturers=4, high_risk=True)
    
    # Dataset 2: 1000 records, 25 machines, 5 manufacturers, high risk
    create_dataset("sample_data_2.csv", num_machines=25, n_days=40, num_manufacturers=5, high_risk=True)

    # Dataset 3: 1800 records, 15 machines, 2 manufacturers, high risk
    create_dataset("sample_data_3.csv", num_machines=15, n_days=120, num_manufacturers=2, high_risk=True)

    print("All datasets generated.")