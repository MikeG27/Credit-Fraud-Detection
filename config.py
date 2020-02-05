import os

# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# RAW
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_DATA = os.path.join(RAW_DIR,"creditcard.csv")

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
FIGURES_FRAUDS_COUNTER = os.path.join(FIGURES_DIR,"Frauds_counter.png")
FIGURES_PCA_COMPONENTS = os.path.join(FIGURES_DIR,"Pca_components.png")
FIGURES_DISTRIBUTION = os.path.join(FIGURES_DIR,"Class_distribution.png")
FIGURES_CORRELATIONS = os.path.join(FIGURES_DIR,"Correlation.png")
FIGURES_TRANSACTIONS_SCATTER = os.path.join(FIGURES_DIR,"Transactions_scatter.png")
FIGURES_TRANSACTIONS_HISTOGRAM = os.path.join(FIGURES_DIR,"Transactions_histogram.png")
FIGURES_TRANSACTIONS_PDF = os.path.join(FIGURES_DIR, "Transactions_pdf.png")
FIGURES_TRANSACTIONS_ECDF = os.path.join(FIGURES_DIR, "Transactions_ecdf.png")

