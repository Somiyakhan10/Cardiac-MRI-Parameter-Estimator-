import pandas as pd

PATHOLOGY_ORDER = ["NOR", "DCM", "HCM", "MINF", "RV"]

PATHOLOGY_COLORS = {
    "NOR": "#2a78d6",
    "DCM": "#008300",
    "HCM": "#e87ba4",
    "MINF": "#eda100",
    "RV": "#1baf7a",
}

PATHOLOGY_NAMES = {
    "NOR": "Normal (Healthy)",
    "HCM": "Hypertrophic Cardiomyopathy",
    "DCM": "Dilated Cardiomyopathy",
    "MINF": "Myocardial Infarction",
    "RV": "Abnormal Right Ventricle",
}

STATUS_COLORS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

def get_patient_record(df: pd.DataFrame, patient_id: str) -> dict:
    row = df[df["patient_id"] == patient_id].iloc[0]
    return row.to_dict()

def confidence_badge(confidence: float, pathology: str) -> tuple:
    if pathology == "NOR":
        return "Healthy Subject - High Confidence", "good"
    if confidence >= 0.78:
        return "Pathological Case - Moderate Confidence", "warning"
    return "Pathological Case - Lower Confidence", "critical"
