import os

# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# RAW
RAW_DIR = os.path.join(DATA_DIR, "raw")

# PREPROCESSED
DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

# MODELS
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_TRAIN_METRICS = os.path.join(MODEL_DIR, "train_metrics.txt")
MODEL_TEST_METRICS = os.path.join(MODEL_DIR, "test_metrics.txt")


# NOTEBOOKS
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")

# REFERENCES
REFERENCES_DIR = os.path.join(ROOT_DIR, "references")

# REPORTS
REPORTS_DIR = os.path.join(ROOT_DIR,"reports")

# FIGURES
FIGURES_DIR = os.path.join(REPORTS_DIR,"figures")

