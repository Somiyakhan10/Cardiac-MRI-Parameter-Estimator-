import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _find_model_file() -> str:
    keras_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
    h5_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5")))
    candidates = keras_files + h5_files
    if candidates:
        return candidates[0]
    return os.path.join(MODELS_DIR, "acdc_parameter_estimator.keras")


MODEL_PATH = os.environ.get("MODEL_PATH", _find_model_file())

ACDC_DATA_PATH = os.environ.get(
    "ACDC_DATA_PATH",
    os.path.join(BASE_DIR, "data", "acdc_challenge_20170617")
)
