import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
N_MACHINES = 50
N_DAYS = 365
DATA_POINTS_PER_DAY = 4

# Component failure rates (failures per 1000 hours)
COMPONENT_FAILURE_RATES = {
    "Engine": 0.1,
    "Hydraulic System": 0.15,
    "Transmission": 0.08,
    "Electrical System": 0.2,
    "Cooling System": 0.12,
}

MACHINE_TYPES = ["Haul Truck", "Excavator", "Dozer", "Grader", "Loader"]
COMPONENTS = list(COMPONENT_FAILURE_RATES.keys())

def generate_machine_data(machine_id, machine_type, start_date, n_days):
    """Generates data for a single machine over a period of time."""
    data = []
    operating_hours = np.random.randint(500, 2000)
    
    for day in range(n_days):
        for _ in range(DATA_POINTS_PER_DAY):
            # Normal operating conditions
            vibration = np.random.normal(5, 1.5)
            temperature = np.random.normal(80, 5)
            pressure = np.random.normal(150, 10)
            
            # Simulate degradation leading to failure
            for component, rate in COMPONENT_FAILURE_RATES.items():
                # Probability of failure in the next hour
                prob_failure = (rate / 1000) * (operating_hours / 1000)

                if np.random.rand() < prob_failure:
                    # Anomaly leading to failure
                    vibration *= np.random.uniform(1.5, 3.0)
                    temperature += np.random.uniform(10, 30)
                    pressure += np.random.uniform(20, 40)
                    
            timestamp = start_date + timedelta(days=day, hours=np.random.randint(0, 24))
            
            data.append([
                machine_id,
                machine_type,
                timestamp,
                vibration,
                temperature,
                pressure,
                operating_hours,
            ])
            operating_hours += 6 # Assume 6 hours of operation per data point

    return pd.DataFrame(data, columns=[
        "MachineID", "MachineType", "Timestamp", "Vibration", 
        "Temperature", "Pressure", "OperatingHours"
    ])

def introduce_failures(df, prediction_window=14):
    """Introduces failures and labels the data accordingly."""
    df["Failure"] = 0
    df["FailedComponent"] = "None"
    
    for machine_id in df["MachineID"].unique():
        machine_df = df[df["MachineID"] == machine_id].copy()
        
        # Determine when failures occur
        failure_points = machine_df.sample(frac=0.01) # 1% of data points are failures
        
        for idx, row in failure_points.iterrows():
            failure_time = row["Timestamp"]
            failed_component = np.random.choice(COMPONENTS)
            
            # Mark the failure point
            df.loc[idx, "FailedComponent"] = failed_component
            
            # Label the data within the prediction window
            time_window = (df["Timestamp"] >= (failure_time - timedelta(days=prediction_window))) & \
                          (df["Timestamp"] < failure_time) & \
                          (df["MachineID"] == machine_id)
            
            df.loc[time_window, "Failure"] = 1

    return df

def main():
    """Main function to generate the full dataset."""
    print("Generating dataset...")
    start_date = datetime(2023, 1, 1)
    all_machine_data = []

    for i in range(N_MACHINES):
        machine_type = np.random.choice(MACHINE_TYPES)
        machine_df = generate_machine_data(f"M{i+1}", machine_type, start_date, N_DAYS)
        all_machine_data.append(machine_df)

    full_df = pd.concat(all_machine_data, ignore_index=True)
    
    # Introduce failures after all data is generated
    full_df = introduce_failures(full_df)

    # Save to CSV
    output_path = "C:\Users\HP\MachineFailurePredictor\data\machine_failure_data.csv"
    full_df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")

if __name__ == "__main__":
    main()