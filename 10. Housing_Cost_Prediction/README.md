# Агенство недвижимости (Housing Cost Prediction)

<div align="center">

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/xndrf/Data_Science_Project/blob/master/10.%20Housing_Cost_Prediction/model_building.ipynb)

</div>

# 

## Цель проекта

**Разработка модели машинного обучения и программного сервиса для точного прогнозирования рыночной стоимости объектов жилой недвижимости. на основе истории предложений.**


<table style="width: 100%; border: none; border-collapse: collapse; margin: 15px 0;">
  <tr>
    <td style="vertical-align: top; padding: 10px; width: 50%;">
      <ul style="padding-left: 20px; margin: 0; line-height: 1.6;">
        <strong>Основные задачи:</strong>
        <li>Выполнить анализ, структуризацию и подготовку данных к EDA;</li>
        <li>Провести EDA, оценить распределение и взаимосвязи ключевых переменных;</li>
        <li>Обогатить данные путём проведения парсинга и создания дополнительных признаков;</li>
        <li>Сформировать гипотезы по влиянию признаков на целевую переменную.</li>
        <li>Обучить и сравнить базовые модели;</li>
        <li>Выполнить оптимизацию гиперпараметров;</li>
        <li>Построить ансамбль моделей;</li>
        <li>Разработать пользовательский веб‑интерфейс;</li>
        <li>Выполнить контейнеризацию и воспроизводимый запуск с использованием Docker.</li>
        <br>
        <strong>Данные:</strong>
        <br>В проекте используются структурированные табличные данные, содержащие:
        <li>характеристики объекта недвижимости (площадь, количество комнат, этажность и т.д.);</li>
        <li>инженерные системы (отопление, охлаждение, парковка);</li>
        <li>городскую и социальную инфраструктуру;</li>
        <li>экономические и временные признаки.</li>
        <br>
        <strong>Дополнительные данные:</strong>
        <li>Выполнен парсинг данный геолокаций по zipcode;</li>
        <li>Найден дополнительный датасет из kaggle по локациям для использования в построении дополнительных признаков.</li>
      </ul>
    </td>
    <td style="vertical-align: top; padding: 10px; text-align: center; width: 5 Newton;">
      <img 
        src="image/read.jpg"
        alt="Иллюстрация процесса"
        style="max-width: 50%; height: auto; border-radius: 8px;"
      >
    </td>
  </tr>
</table>

## Этапы работы

### 1. Предобработка данных:

- приведение показателей к единому виду;
- распаковка вложенных структур (создание колонок: year_built, remodeled_year, heating, cooling, parking, lotsize, price_per_sqft);
- группировка категориальных значений;
- извлечение признаков из поля schools (schools_mean_distance, schools_min_distance, schools_count);
- объединение дублирующих признаков (MlsId/mls-id, PrivatePool/private pool);
- парсинг ZIP‑кодов для получения координат.

### 2. Анализ и обогащение:

- логарифмическое преобразование целевой переменной для стабилизации дисперсии;
- отбор значимых признаков через Mutual Information;
- создание пайплайна предобработки с ColumnTransformer.

### 3. Моделирование:
- тестирование линейных моделей (Ridge, Lasso, ElasticNet) и ансамблевых (XGBoost, CatBoost, LightGBM, Random Forest);
- оптимизация гиперпараметров через RandomizedSearchCV и Optuna;
- построение стекинга с мета‑моделью Ridge.

### 4. Валидация:
- оценка по метрикам: $R^2$, MAE, RMSLE, MAPE;
- анализ времени обучения и инференса;
- исследование корреляции ошибок базовых моделей.

### 5. Результаты
**Качество финальной модели (стекинг XGBoost + CatBoost + LightGBM + Ridge):**
- **$R^2$=0,9364** (в логарифмической шкале) — объясняет >93% вариации целевой переменной;
- **$R^2$=0,8992** (в исходной шкале) — устойчивость после обратного преобразования;
- **RMSLE = 0,2169** — контроль относительных ошибок;
- **MAE ≈ 73 693** (в абсолютных значениях);
- **MAPE = 14,66%**.

### Ключевые достижения:
- логарифмирование таргета снизило гетероскедастичность;
- отбор признаков через Mutual Information уменьшил шум;
- стекинг усилил сильные стороны градиентных бустингов.

---

### Финальная модель

**Окончательной моделью проекта является ансамбль моделей типа Stacking**, сохранённый в файле:

```
models/final_stacking_model.pkl
```

#### Архитектура ансамбля

* **1 уровень:** набор базовых моделей (CatBoost, XGBoost, LightGBM, Random Forest)
* **2 уровень:** meta‑модель (CatBoost, XGBoost, LightGBM)

Файл:

```
models/final_meta_model.pkl
```

---

## Веб‑приложение
Реализован интерактивный веб-интерфейс Streamlit для инференса модели, включающий контроль входных признаков, описания параметров и механизм логирования предсказаний»
Как запустить Streamlit (`app.py`)

1. Открыть терминал 
2. Установить python -m pip install streamlit
3. Перейти в папку проекта (Например cd "D:\code\IDE_XNDRF\Data_Science_Project\10. Housing_Cost_Prediction")
4. Выполняем запуск python -m streamlit run app.py
5. Открывается окно браузера и мы делаем предсказания
![Dashboard demo](https://github.com/xndrf/Data_Science_Project/blob/master/10.%20Housing_Cost_Prediction/image/terminal.gif)

Функциональные возможности:

* интерактивный ввод параметров объекта недвижимости;
* получение прогноза стоимости в реальном времени;
* логирование всех предсказаний в CSV‑файл;
* обработка ошибок и контроль корректности данных.

![Dashboard demo](https://github.com/xndrf/Data_Science_Project/blob/master/10.%20Housing_Cost_Prediction/image/lokal.gif)

Инференс выполняется через модуль:

```
src/predict.py
```

---

## Использование Docker

<table style="width: 100%; border: none; border-collapse: collapse; margin: 15px 0;">
  <tr>
    <td style="vertical-align: top; padding: 10px; width: 50%;">
      <ul style="padding-left: 20px; margin: 0; line-height: 1.6;">
    Проект полностью контейнеризирован, что обеспечивает <strong>воспроизводимость результатов</strong> и упрощает проверку работы модели.
<br>Установка Docker<br>
       Необходимо установить <strong>Docker Desktop</strong>:

🔗 [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

Поддерживаемые ОС:
* Windows (WSL2);
* macOS;
* Linux.
      </ul>
    </td>
    <td style="vertical-align: top; padding: 10px; text-align: center; width: 5 Newton;">
      <img 
        src="image/docker.jpg"
        alt="Иллюстрация процесса"
        style="max-width: 40%; height: auto; border-radius: 8px;">
    </td>
  </tr>
</table>


Проверка установки:

```bash
docker --version
```

---

### Получение проекта

#### Качаем GitHub

```bash
git clone https://github.com/xndrf/housing-cost-prediction.git
cd housing-cost-prediction
```

---

### Сборка Docker‑образа

```bash
docker build -t housing-price-app .
```

В процессе сборки:

* устанавливаются зависимости;
* копируются только необходимые файлы (через `.dockerignore`);
* загружается обученная модель;
* подготавливается Streamlit‑приложение.

---

### Запуск приложения

```bash
docker run -p 8501:8501 housing-price-app
```

После запуска приложение доступно по адресу:

[http://localhost:8501](http://localhost:8501)


---
## Структура проекта

```

├── base_models/                        # Базовые модели 
├── catboost_info/                      # catboost 
├── data/                               # Исходные и обработанные данные
├── image/                              # Мультимедийные файлы
├── logs/                               # Логи предсказаний
├── models/                             # Финальные модели и артефакты
├── reports/                            # Дополнительная папка для отчетов и визуализации
├── src/                                # Логика препроцессинга и инференса
├── .dockerignore                       # Исключение лишних файлов
├── .gitignore                          # Исключение лишних файлов
├── app.py                              # Веб-интерфейс (Streamlit)
├── data_preparation.ipynb              # Ноутбук предварительной подготовки данных
├── Dockerfile                          # Конфигурация Docker
├── exploratory_data_analysis.ipynb     # Ноутбук EDA
├── model_building.ipynb                # Ноутбук построения модели
├── pred_test.ipynb                     # Предсказание модели внутри ноутбука
└── requirements.txt                    # Зависимости
```

---

## Рекомендации для развития


1. Доработка данных:

- парсинг дополнительных признаков: средние зарплаты по районам, сейсмическая активность, климатические данные, плотность застройки;
- кластеризация районов по координатам;
- анализ выбросов и крайних ценовых диапазонов.

2. Оптимизация модели:

- разбиение датасета на ценовые сегменты;
- тестирование других мета‑моделей для стекинга (например, ElasticNet).

3. Внедрение:

- согласование с заказчиком требуемой точности и времени обработки;
- оценка вычислительных ресурсов для продакшна;
- учёт специфики рыночных сегментов.
