import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """Загрузка конфигурации"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Конфигурация загружена {config_path}")
        return config
    
    # На ВСЯКИЙ СЛУЧАЙ вдруг кто скопирует не правильно и забудет создать файл конфигурации
    
    except FileNotFoundError:
        print(f"Файл конфигурации {config_path} не найден!")
        print("Используем конфигурацию по умолчанию...")
        return {
            'data': {
                'max_texts_count': 100000,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_state': 42
            },
            'tokenization': {
                'max_len': 64,
                'model_name': 'gpt2'
            },
            'lstm': {
                'embed_dim': 256,
                'hidden_dim': 512,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'weight_decay': 0.01
            },
            'training': {
                'batch_size': 128,
                'num_epochs': 10,
                'clip_grad_norm': 1.0
            },
            'generation': {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.9,
                'max_length': 30
            },
            'evaluation': {
                'rouge_examples': 50,
                'test_examples': 5
            }
        }

def load_and_split_data(config, raw_data_path="data/cleaned_tweets.csv"):
    """
    Загрузка и разделение данных
    
    Args:
        config: конфигурация
        raw_data_path: путь к исходному файлу с данными
    
    Returns:
        train_texts, val_texts, test_texts
    """
    print(f"\nЧитаем датасет {raw_data_path}...")
    
    # Загружаем данные
    df = pd.read_csv(raw_data_path, encoding="utf-8-sig", sep="\t")
    text_column = 'clean_text'
    
    # Удаляем пустые строки
    df.dropna(subset=[text_column], inplace=True)
    
    # Берем нужное количество текстов
    texts = df[text_column].tolist()
    original_count = len(texts)
    texts = texts[:config['data']['max_texts_count']]
    
    print(f"\nВсего твитов: {original_count}")
    print(f"Используем: {len(texts)}")
    
    # Разделяем на train/val/test
    # train (80%)
    train_texts, temp_texts = train_test_split(
        texts,
        test_size=config['data']['val_split'] + config['data']['test_split'],  # 0.2
        random_state=config['data']['random_state']
    )
    
    # val и test (по 10%)
    val_texts, test_texts = train_test_split(
        temp_texts,
        test_size=config['data']['test_split'] / (config['data']['val_split'] + config['data']['test_split']),
        random_state=config['data']['random_state']
    )
    
    print(f"\nРазмеры выборок:")
    print(f"Train: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")
    
    # Сохраняем разделенные данные для последующего использования
    os.makedirs("data", exist_ok=True)
    
    pd.DataFrame({'text': train_texts}).to_csv("data/train.csv", index=False)
    pd.DataFrame({'text': val_texts}).to_csv("data/val.csv", index=False)
    pd.DataFrame({'text': test_texts}).to_csv("data/test.csv", index=False)
    
    print(f"\nРазделенные датасеты сохранены:")
    print(f"data/train.csv")
    print(f"data/val.csv")
    print(f"data/test.csv")
    
    return train_texts, val_texts, test_texts

def load_split_data():
    """
    Загрузка уже разделенных данных (если они были сохранены ранее)
    
    Returns:
        train_texts, val_texts, test_texts
    """
    # Проверяем существование файлов
    required_files = ["data/train.csv", "data/val.csv", "data/test.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Файлы {missing_files} не найдены!")
        print("Сначала нужно выполнить load_and_split_data()")
        return None, None, None
    
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")
    
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    
    print(f"\nЗагружены разделенные датасеты:")
    print(f"Train: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")
    
    return train_texts, val_texts, test_texts

def prepare_data_pipeline(config, force_reload=False):
    """
    Полный пайплайн подготовки данных
    
    Args:
        config: конфигурация
        force_reload: принудительно перезагрузить и разделить данные заново
    
    Returns:
        train_texts, val_texts, test_texts
    """
    if force_reload:
        print("Принудительная перезагрузка данных")
        return load_and_split_data(config)
    
    # Пробуем загрузить уже разделенные данные
    train_texts, val_texts, test_texts = load_split_data()
    
    # Если не получилось, загружаем и разделяем заново
    if train_texts is None:
        print("Разделенные данные не найдены. Выполняем разделение")
        return load_and_split_data(config)
    
    return train_texts, val_texts, test_texts

# Для обратной совместимости
__all__ = ['load_config', 'load_and_split_data', 'load_split_data', 'prepare_data_pipeline']