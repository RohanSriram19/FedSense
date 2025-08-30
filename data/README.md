# FedSense Data Directory

This directory contains datasets for federated time-series anomaly detection.

## Expected Data Schema

All CSV files should follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Timestamp of the measurement |
| `hr` | float | Heart rate in beats per minute (BPM) |
| `accel_x` | float | X-axis acceleration (m/s²) |
| `accel_y` | float | Y-axis acceleration (m/s²) |
| `accel_z` | float | Z-axis acceleration (m/s²) |
| `label` | int | Binary label (0=normal, 1=anomaly) |

Example:
```csv
timestamp,hr,accel_x,accel_y,accel_z,label
2024-01-01 00:00:00.000,75.2,0.1,0.2,9.8,0
2024-01-01 00:00:00.020,75.5,0.0,0.1,9.9,0
2024-01-01 00:00:00.040,95.8,2.1,1.5,11.2,1
```

## Data Sources

### Synthetic Data (Generated)
The system can generate synthetic wearable sensor data for testing and development:

```bash
# Generate federated data splits
make data
```

This creates:
- `client_0/data.csv` through `client_N/data.csv` - Individual client datasets
- `global_test.csv` - Global test set for evaluation

### Public Datasets (Optional)

You can use public wearable datasets such as:

1. **WESAD (Wearable Stress and Affect Detection)**
   - Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
   - Contains physiological signals during stress/baseline conditions
   - Map stress conditions to anomaly labels

2. **PAMAP2 Physical Activity Monitoring**
   - Download from: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
   - Contains accelerometer and heart rate data during activities
   - Map high-intensity activities to anomaly labels

### Data Preprocessing

For custom datasets:

1. **Sampling Rate**: Ensure data is sampled at 50 Hz or resample appropriately
2. **Window Length**: Default window is 250 samples (5 seconds at 50 Hz)
3. **Feature Scaling**: Features are automatically standardized during training
4. **Missing Values**: Handle missing values before feeding to the system

### Federated Data Distribution

Client data is distributed in a non-IID manner using Dirichlet distribution with concentration parameter α=0.1 (configurable). This creates realistic federated scenarios where:

- Each client has different proportions of normal vs. anomalous data
- Activity patterns vary across clients
- Some clients may have very few anomalies (class imbalance)

### Data Generation Parameters

When generating synthetic data:

- **Base Heart Rate**: 75 ± 8 BPM (normal distribution)
- **Anomaly Heart Rate**: 1.3-1.8x base rate
- **Base Acceleration**: ~1g with small variations
- **Anomaly Acceleration**: 2-4x base magnitude
- **Anomaly Rate**: 10-15% of samples
- **Window Overlap**: 80% (stride = 20% of window_len)

## Directory Structure

```
data/
├── README.md                 # This file
├── client_0/
│   └── data.csv             # Client 0 data
├── client_1/
│   └── data.csv             # Client 1 data
├── ...
├── client_N/
│   └── data.csv             # Client N data
└── global_test.csv          # Global test set
```

## Quality Checks

Before training, verify your data:

1. **Schema Validation**: All required columns present
2. **Timestamp Ordering**: Timestamps are monotonically increasing
3. **Sampling Rate**: Consistent time intervals between samples
4. **Value Ranges**: HR (40-200 BPM), Accelerations (-20 to +20 m/s²)
5. **Label Balance**: At least 5-20% anomalies for meaningful detection

Use the data validation utilities:

```python
from fedsense.datasets import validate_data_quality
validate_data_quality("data/client_0/data.csv")
```
