import os
import glob

def _find_model_file() -> str:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        return os.path.join(models_dir, "acdc_model_compatible.h5")
    
    keras_files = sorted(glob.glob(os.path.join(models_dir, "*.keras")))
    h5_files = sorted(glob.glob(os.path.join(models_dir, "*.h5")))
    candidates = keras_files + h5_files
    
    if candidates:
        return candidates[0]
    
    return os.path.join(models_dir, "acdc_model_compatible.h5")

MODEL_PATH = os.environ.get("MODEL_PATH", _find_model_file())

ACDC_DATA_PATH = os.environ.get(
    "ACDC_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "acdc_challenge_20170617")
)
