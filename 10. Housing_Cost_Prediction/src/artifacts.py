import os
import sys
import joblib
from pathlib import Path

from src.encoders import BinaryEncoderWrapper

# Регистрируем кастомный класс для pickle
sys.modules['__main__.BinaryEncoderWrapper'] = BinaryEncoderWrapper


def get_project_root() -> Path:
    """
    Корень проекта:
    - Docker / PROD: через переменную окружения PROJECT_ROOT
    - Локально: на два уровня выше текущего файла
    """
    env_path = os.getenv("PROJECT_ROOT")
    if env_path:
        return Path(env_path).resolve()

    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = PROJECT_ROOT / "base_models"


def load_artifacts():
    # preprocessing
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    selected_features = joblib.load(MODELS_DIR / "selected_features.pkl")

    # base models
    xgb = joblib.load(BASE_MODELS_DIR / "XGBoost_Optuna.pkl")
    cat = joblib.load(BASE_MODELS_DIR / "CatBoost_Optuna.pkl")
    lgb = joblib.load(BASE_MODELS_DIR / "LightGBM_Optuna.pkl")

    # meta model
    meta_model = joblib.load(MODELS_DIR / "final_meta_model.pkl")

    return preprocessor, selected_features, (xgb, cat, lgb), meta_model
