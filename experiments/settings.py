import os
from datetime import datetime

# --- System Settings ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results/experiment_{TIMESTAMP}"
os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)

# --- Hyperparameter Search Space ---
# Note: Defined as dictionaries so our RandomSearcher (and future GeneticSearcher) can read them!
CONFIG_SPACE = {
    "learning_rate": {'type': 'continuous', 'range': (1e-4, 1e-1)},
    "hidden_units": {'type': 'discrete', 'values': [32, 64, 128, 256, 512]},
    "dropout_rate": {'type': 'continuous', 'range': (0.1, 0.7)},
    "batch_size": {'type': 'discrete', 'values': [32, 64, 128, 256]}
}

# --- Search Settings ---
NUM_EVALUATIONS = 20