import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def generate_triad_dataset(n_samples=100, out_dir="data"):
    """
    Generates synthetic ECG signals and metadata compatible with 
    the TRIAD-ECG dataset.py logic.
    """
    waveforms_dir = os.path.join(out_dir, "waveforms")
    os.makedirs(waveforms_dir, exist_ok=True)
    
    records = []
    
    print(f"Generating {n_samples} synthetic ECG files...")
    
    # Time axis for a 10-second ECG at 500Hz
    t = np.linspace(0, 10, 5000)

    for i in tqdm(range(n_samples)):
        file_id = f"ecg_record_{i:05d}"
        filename = f"{file_id}.npy"
        
        # 1. Simulate physiological signals
        # Heart rate between 60-95 bpm
        hr = np.random.uniform(60, 95)
        # Base QRS-like pulse
        base_signal = np.sin(np.pi * (hr/60) * t)**20 
        
        # Create 12 leads (Rows=5000, Cols=12)
        # dataset.py will transpose this to (12, 5000) automatically
        data = np.zeros((5000, 12))
        for lead in range(12):
            lead_scale = np.random.uniform(0.5, 1.5)
            noise = np.random.normal(0, 0.05, 5000)
            data[:, lead] = (base_signal * lead_scale) + noise
        
        # Save as individual .npy file
        np.save(os.path.join(waveforms_dir, filename), data.astype(np.float32))
        
        # 2. Assign Metadata (Labels and Splits)
        # Probability of HFpEF increases with simulated heart rate for the demo
        split = np.random.choice(["train", "valid", "test"], p=[0.7, 0.15, 0.15])
        label = 1 if (hr > 85 or np.random.rand() > 0.8) else 0
        
        records.append({
            "record_name": file_id,
            "HFpEF": label,
            "split": split
        })

    # Save the label CSV in the data folder
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    
    print(f"\nSuccess!")
    print(f"- Waveforms saved to: {waveforms_dir}/")
    print(f"- Metadata saved to: {out_dir}/labels.csv")

if __name__ == "__main__":
    generate_triad_dataset()
