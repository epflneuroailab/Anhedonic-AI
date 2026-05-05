#!/usr/bin/env python3
"""
Pipeline to run extraction scripts in sequence:
1. extract_activations.py
2. extract_neurons.py  
3. target_layers.py
"""

import subprocess
import sys

EXTRACTION_SCRIPT = "extract_activations.py"
NEURON_EXTRACTION_SCRIPT = "extract_neurons.py"
TARGET_LAYERS_SCRIPT     = "target_layers.py"

# Run extract_activations.py
print("Running extract_activations.py...")
result = subprocess.run([sys.executable, EXTRACTION_SCRIPT])
if result.returncode != 0:
    print("extract_activations.py failed!")
    sys.exit(1)

# Run extract_neurons.py
print("\nRunning extract_neurons.py...")
result = subprocess.run([sys.executable, NEURON_EXTRACTION_SCRIPT])
if result.returncode != 0:
    print("extract_neurons.py failed!")
    sys.exit(1)

# Run target_layers.py
print("\nRunning target_layers.py...")
result = subprocess.run([sys.executable, TARGET_LAYERS_SCRIPT])
if result.returncode != 0:
    print("target_layers.py failed!")
    sys.exit(1)

print("\nAll scripts completed successfully!")