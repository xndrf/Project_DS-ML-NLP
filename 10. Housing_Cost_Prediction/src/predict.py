import sys
import numpy as np
import pandas as pd

# импорт кастомных классов
from src.encoders import BinaryEncoderWrapper
from src.artifacts import load_artifacts



# Кэш артефактов (для Docker)

_ARTIFACTS = None


def _get_artifacts():
    """
    Лениво загружает и кэширует артефакты модели.
    Используется для предотвращения повторной загрузки
    тяжелых .pkl файлов при каждом запросе.
    """
    global _ARTIFACTS

    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()

    return _ARTIFACTS


def predict_new_data(input_df: pd.DataFrame) -> float:
    """
    Выполняет инференс ансамблевой модели:
    1. Препроцессинг входных данных
    2. Отбор признаков
    3. Предсказания базовых моделей
    4. Meta-предсказание итоговой стоимости

    Parameters
    ----------
    input_df : pd.DataFrame
        Входные признаки одного объекта недвижимости

    Returns
    -------
    float
        Прогнозируемая стоимость недвижимости
    """

    preprocessor, selected_features, base_models, meta_model = _get_artifacts()

    X_prep = preprocessor.transform(input_df)
    X_prep = pd.DataFrame(
        X_prep,
        columns=preprocessor.get_feature_names_out()
    )

    missing_features = set(selected_features) - set(X_prep.columns)
    if missing_features:
        raise ValueError(
            f"Отсутствуют признаки после препроцессинга: {missing_features}"
        )
        
    X_final = X_prep[selected_features]


    X_meta = np.column_stack([
        model.predict(X_final) for model in base_models
    ])

    y_pred_log = meta_model.predict(X_meta)

    return float(np.exp(y_pred_log)[0])
