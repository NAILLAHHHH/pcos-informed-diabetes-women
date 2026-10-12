"""Central configuration for reproducible experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_FILENAME = "diabetes_012_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_012"
SEX_COL = "Sex"
FEMALE_CODE = 0  # In this cleaned BRFSS indicators file: 0 = female, 1 = male

CLASS_NAMES = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the remaining train split
N_JOBS = -1

# Features available in the cleaned BRFSS diabetes indicators CSV
FEATURE_COLS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
    "Education",
    "Income",
]

# Metabolic / lifestyle features emphasized in PCOS–diabetes literature (motivation only)
PCOS_PHENOTYPE_FEATURES = [
    "BMI",
    "HighBP",
    "HighChol",
    "PhysActivity",
    "GenHlth",
    "Age",
]
