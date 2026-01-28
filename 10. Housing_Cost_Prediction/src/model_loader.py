import os
import joblib

DEFAULT_MODEL_PATH = "/models/final_meta_model.pkl"

def load_model():
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please mount models directory and/or set MODEL_PATH."
        )

    return joblib.load(model_path)
