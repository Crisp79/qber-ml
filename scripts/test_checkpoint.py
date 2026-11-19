#!/usr/bin/env python3
"""Test checkpoint/resume functionality."""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, 'E:/Coding/QBER-ML/src')

from scripts.generate_bb84_dataset import (
    CHECKPOINT_FILE, load_checkpoint, save_checkpoint, session_key
)

output_path = Path("E:/Coding/QBER-ML/data/bb84")

# Create a checkpoint with 4 completed sessions (simulating interruption)
checkpoint = {
    "completed_sessions": ["0_0", "0_1", "1_0", "1_1"],
    "session_results": {},
    "next_session_id": 4,
    "total_sessions": 16,
    "start_time": time.time() - 100,
    "elapsed_time": 100,
    "workers": 2,
}

# Save checkpoint
save_checkpoint(Path("E:/Coding/QBER-ML/data/bb84"), checkpoint)
print("Created checkpoint with 4 completed sessions")

# Now run the generation script which should resume from checkpoint
print("Running generation with resume...")

# The generate script should detect the checkpoint and resume
# We'll just verify the checkpoint loading works
loaded = load_checkpoint(Path("E:/Coding/QBER-ML/data/bb84"))
print(f"Loaded checkpoint: {len(loaded.get('completed_sessions', []))} sessions completed")
print(f"Completed: {loaded.get('completed_sessions', [])}")

# Test session_key function
print(f"\nSession keys:")
for class_idx in range(2):
    for i in range(4):
        print(f"  {session_key(class_idx, i, 4)}")

print("\nCheckpoint test complete!")