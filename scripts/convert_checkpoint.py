#!/usr/bin/env python3
"""Convert checkpoint JSON to CSV dataset."""

import json
import pandas as pd
from pathlib import Path

def checkpoint_to_csv(checkpoint_path, output_csv):
    """Convert checkpoint JSON to CSV dataset."""
    
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    session_results = checkpoint.get('session_results', {})
    
    if not session_results:
        print("No session results in checkpoint!")
        return
    
    # Convert to list of rows
    rows = []
    for i, (session_key, features) in enumerate(session_results.items()):
        row = features.copy()
        row['session_key'] = session_key
        row['session_id'] = i  # Assign sequential ID
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by class_index then session_id for consistency
    if 'class_index' in df.columns:
        df = df.sort_values(['class_index', 'session_id']).reset_index(drop=True)
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Converted {len(df)} sessions to {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['class_label'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "data/bb84/bb84_dataset_checkpoint.json"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/bb84/bb84_temporal_dataset_recovered.csv"
    
    df = checkpoint_to_csv(checkpoint_path, output_csv)