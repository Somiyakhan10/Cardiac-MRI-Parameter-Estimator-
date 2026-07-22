import os

# Path to the trained model
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "acdc_parameter_estimator.keras")
)

# Optional: ACDC dataset path (kept for future use, not required)
ACDC_DATA_PATH = os.environ.get(
    "ACDC_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "acdc_challenge_20170617")
)