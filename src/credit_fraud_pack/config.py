from pathlib import Path

RANDOM_SEED = 4
TEST_SIZE = 0.2

TARGET_COL = "Class"

# Paths

# This is two levels down from the root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Models
MODELS_DIR = ROOT_DIR / "models"

# Outputs and figures
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_COMPARISON_FILE = OUTPUTS_DIR / "model_comparison.csv"

# Cost-matrix constants
COST_FALSE_NEGATIVE = -1     # Missed fraud: money actually lost
COST_FALSE_POSITIVE = -2    # Wrongly blocked legit transaction: incurred investigation costs

# Feature groups
SCALE_COLS = ["Time", "Amount"]             # Need scaling, not part of PCA
PCA_COLS = [f"V{i}" for i in range(1, 29)]  # V1-V28, already PCA-transformed
FEATURE_COLS = SCALE_COLS + PCA_COLS        # The model's input contract

