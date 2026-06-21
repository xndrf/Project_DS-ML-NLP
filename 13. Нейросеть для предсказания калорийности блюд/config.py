import torch

class Config:
    # Пути
    DISH_PATH = "data/clear_dish.csv"
    INGR_PATH = "data/clear_ingredients.csv"
    IMAGE_ROOT = "data/images"
    SAVE_PATH = "models/best_model.pth"

    # Модели
    TEXT_MODEL_NAME = "microsoft/deberta-v3-small"
    IMAGE_MODEL_NAME = "convnext_tiny"

    # Разморозка
    TEXT_MODEL_UNFREEZE = "encoder.layer.[4-5]"
    IMAGE_MODEL_UNFREEZE = "stages.[2-3]|head"

    # Гиперпараметры
    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 5e-5
    HEAD_LR = 5e-4
    EPOCHS = 15
    DROPOUT = 0.4
    HIDDEN_DIM = 512
    MAX_TEXT_LEN = 128

    # Оверсемплинг
    USE_OVERSAMPLING = True
    OVERSAMPLING_STRATEGY = "calorie_based"
    
    # Фичи
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_LOG_TARGET = True
    USE_MASS_FEATURE = True
    USE_CALORIE_DENSITY = True