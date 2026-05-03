"""Project-wide configuration: paths, the random seed and column groups.

Keeping these in one place means the notebooks, the tests and the Streamlit app
all agree on where the data lives and which columns are treated how.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# One seed for every split / model so results are reproducible.
RANDOM_STATE = 42

# --- raw schema ----------------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"

# Columns that arrive as text but are really categorical.
RAW_CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric columns present in the raw file.
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
