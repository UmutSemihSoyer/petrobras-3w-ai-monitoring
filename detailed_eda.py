import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

DATASET_DIR = "dataset"
OUTPUT_DIR = "reports_eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes_info = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line"
}

print("Analyzing Petrobras 3W Dataset...")

records = []
for c in range(10):
    folder_path = os.path.join(DATASET_DIR, str(c))
    if not os.path.exists(folder_path):
        continue
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    for f in files:
        filename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024.0
        
        # Source type detection: WELL-XXXXX (Real), SIMULATED, DRAWN
        if filename.startswith("WELL"):
            source_type = "Real Well"
            well_id = filename.split("_")[0]
        elif filename.startswith("SIMULATED"):
            source_type = "Simulated"
            well_id = "Simulated"
        elif filename.startswith("DRAWN"):
            source_type = "Hand-Drawn"
            well_id = "Hand-Drawn"
        else:
            source_type = "Other"
            well_id = "Unknown"
            
        records.append({
            "class_id": c,
            "class_name": classes_info[c],
            "file_name": filename,
            "file_path": f,
            "source_type": source_type,
            "well_id": well_id,
            "size_kb": size_kb
        })

df_all = pd.DataFrame(records)
print(f"Total instances found: {len(df_all)}")

# Grouping by Class & Source Type
df_pivot = pd.crosstab(
    df_all['class_id'].map(lambda c: f"{c}: {classes_info[c]}"), 
    df_all['source_type']
)
print("\nInstance Distribution by Class & Data Source:")
print(df_pivot)

# Plot 1: Instance Counts per Event Class
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
df_pivot.plot(kind='barh', stacked=True, ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
ax.set_title("Petrobras 3W Dataset: Instance Count by Event Class & Source", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Number of Instances", fontsize=11)
ax.set_ylabel("Event Class", fontsize=11)
plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(plot1_path)
plt.close()
print(f"Saved plot: {plot1_path}")

# Sensor inspection on sample files from each class
print("\nInspecting Sensor Availability across Sample Files...")
sample_features_stats = []
for c in range(10):
    class_files = df_all[df_all['class_id'] == c]['file_path'].tolist()
    if not class_files:
        continue
    sample_file = class_files[0]
    df_s = pd.read_parquet(sample_file)
    non_null_counts = df_s.notnull().sum()
    total_len = len(df_s)
    
    active_sensors = [col for col in df_s.columns if col not in ['class', 'state'] and non_null_counts[col] > 0]
    
    sample_features_stats.append({
        "Class ID": c,
        "Class Name": classes_info[c],
        "Sample File": os.path.basename(sample_file),
        "Duration (points)": total_len,
        "Active Sensors": len(active_sensors),
        "Active Sensor List": ", ".join(active_sensors[:6]) + ("..." if len(active_sensors) > 6 else "")
    })

df_sensor_summary = pd.DataFrame(sample_features_stats)
print("\nSample File Feature Availability:")
print(df_sensor_summary.to_string(index=False))

# Plot 2: Time Series Visualization of a Transient Event (Class 2: Spurious Closure of DHSV or Class 8)
sample_event_file = None
for f in df_all[df_all['class_id'] == 2]['file_path']:
    sample_event_file = f
    break

if sample_event_file:
    df_ev = pd.read_parquet(sample_event_file)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=150)
    
    # Plot Pressure sensors
    p_cols = [c for c in ['P-PDG', 'P-TPT', 'P-MON-CKP', 'P-JUS-CKP'] if c in df_ev.columns and df_ev[c].notnull().any()]
    for col in p_cols:
        axes[0].plot(df_ev.index, df_ev[col] / 1e5, label=col + " (bar)") # convert Pa to bar
    axes[0].set_ylabel("Pressure (bar)", fontweight='bold')
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"3W Sensor Signal Profile - Event: {os.path.basename(sample_event_file)}", fontsize=13, fontweight='bold')

    # Plot Temperature sensors
    t_cols = [c for c in ['T-PDG', 'T-TPT', 'T-MON-CKP', 'T-JUS-CKP'] if c in df_ev.columns and df_ev[c].notnull().any()]
    for col in t_cols:
        axes[1].plot(df_ev.index, df_ev[col], label=col + " (°C)")
    axes[1].set_ylabel("Temperature (°C)", fontweight='bold')
    axes[1].legend(loc="upper right")

    # Plot Labels (class & state)
    if 'class' in df_ev.columns:
        axes[2].plot(df_ev.index, df_ev['class'], label='Target Class Label', color='crimson', linewidth=2)
    axes[2].set_ylabel("Event Label", fontweight='bold')
    axes[2].set_xlabel("Time", fontweight='bold')
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "sample_sensor_profile.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved sample event plot: {plot2_path}")

print("\nEDA Completed successfully!")
