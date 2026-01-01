from pathlib import Path

# CONFIGURATION (Please set these TWO paths MAUALLY!)

# 1. PROJECT ROOT
# The folder containing 'splits', 'checkpoints' etc.
# We use automatic detection by default, but you should change it manually if required.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. DATASET ROOT
# The folder containing 'EuroSAT_RGB' and 'EuroSAT_MS'.
# !!! PLEASE UPDATE THIS PATH TO YOUR LOCAL DATASET LOCATION !!!
DATASETS_ROOT = PROJECT_ROOT

# INTERNAL PATHS (Do not edit below)
SPLITS_ROOT = PROJECT_ROOT / "splits"

# Dataset paths
RGB_DATASET_ROOT = DATASETS_ROOT / "EuroSAT_RGB"
MS_DATASET_ROOT = DATASETS_ROOT / "EuroSAT_MS"

# Fixed seed
SEED = 3719704
