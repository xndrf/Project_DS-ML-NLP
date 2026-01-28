import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime


# Настройки проекта
from src.predict import predict_new_data

LOG_PATH = "logs/predictions_log.csv"
os.makedirs("logs", exist_ok=True)

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
    layout="centered"
)

st.title("🏠 Прогноз стоимости недвижимости")
st.markdown(
    "Веб-сервис для оценки стоимости жилой недвижимости на основе ансамбля моделей машинного обучения."
)

st.divider()


# Форма ввода данных
with st.form("prediction_form"):

    st.subheader("📍 Общая информация")

    state = st.selectbox(
        "Штат",
        ["FL", "TX", "CA", "NC", "NY", "TN", "WA", "OH", "NV", "IL", "CO", "GA",
         "PA", "DC", "AZ", "MI", "IN", "OR", "MA", "OTHER"],
        help="Штат США, в котором расположен объект недвижимости"
    )

    group_status = st.selectbox(
        "Статус объявления",
        ["Active", "Other", "Auctions", "New", "Waiting for confirmantion", "Signing"],
        help="Текущий статус объявления на рынке недвижимости"
    )

    group_property = st.selectbox(
        "Тип недвижимости",
        ["Single-Family", "Condominium/Apartment", "Townhouse", "Multi-Family",
         "Traditional", "Contemporary/Modern", "Mobile/Manufactured", "Co-op",
         "Ranch", "Other"],
        help="Классификация объекта недвижимости по типу застройки"
    )

    st.subheader("🏗️ Характеристики объекта")

    area = st.number_input(
        "Площадь дома (sqft)",
        min_value=550.0,
        max_value=8000.0,
        value=2000.0,
        help="Жилая площадь объекта недвижимости в квадратных футах"
    )

    lotsize_sqft = st.number_input(
        "Площадь участка (sqft)",
        min_value=630.0,
        max_value=500000.0,
        value=8000.0,
        help="Площадь земельного участка, на котором расположен объект"
    )

    baths = st.selectbox(
        "Количество ванных комнат",
        ["0–1", "1.5", "1.75", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0", "7.5", "8.0", "10+", "no_data", "other"],
        help="Количество ванных комнат (сгруппированное представление)"
    )

    beds = st.selectbox(
        "Количество спален",
        ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9+", "no data"],
        help="Количество спален в объекте недвижимости"
    )

    stories_grouped = st.selectbox(
        "Этажность",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+", "20+"],
        help="Количество этажей в здании"
    )

    pool_bool = st.checkbox(
        "Наличие бассейна",
        help="Флаг наличия бассейна на территории объекта"
    )

    fireplace_clear = st.checkbox(
        "Наличие камина",
        help="Флаг наличия камина в доме"
    )

    remodeled_year = st.checkbox(
        "Проведена реновация",
        help="Признак проведения капитального ремонта или реновации"
    )

    st.subheader("🔥 Инженерные системы")

    group_heating = st.selectbox(
        "Отопление",
        ["Forced Air Systems", "Electric Systems", "Gas Systems", "Heat Pump Systems",
         "Radiant Systems", "Central Systems", "No Heating", "Other Systems", "Unknown"],
        help="Тип системы отопления объекта"
    )

    group_cooling = st.selectbox(
        "Охлаждение",
        ["Central Systems", "No Cooling", "Heat Pump Systems", "Evaporative Systems",
         "Window/Wall Units", "Refrigeration Systems", "Other Systems", "Unknown"],
        help="Тип системы кондиционирования или охлаждения"
    )

    group_parking = st.selectbox(
        "Парковка",
        ["Attached Garage", "Detached Garage", "Carport", "Driveway", "On Street",
         "Off Street", "Garage Door Opener", "No Parking", "Other parking", "Unknown"],
        help="Тип парковочного пространства, доступного для объекта"
    )

    st.subheader("🏙️ Городская среда")

    city_category = st.selectbox(
        "Категория города",
        ["<100", "100–250", "250–500", "500–1000", "1000–1500", "1500–2500",
         "Houston", "Miami", "Dallas", "Chicago", "Los Angeles", "Other"],
        help="Категория города или агломерации по размеру и значимости"
    )

    city_class = st.selectbox(
        "Тип населённого пункта",
        ["Metropolises", "Large cities", "Mid-size cities", "Small towns", "Micro-settlements"],
        help="Классификация населённого пункта по масштабу"
    )

    density_class = st.selectbox(
        "Плотность застройки",
        ["Low", "Medium", "High", "Very High"],
        help="Уровень плотности застройки в районе объекта"
    )

    from_center = st.number_input(
        "Расстояние до центра города",
        min_value=0.0,
        max_value=5000.0,
        value=5.0,
        help="Расстояние от объекта до центра города"
    )

    city_military = st.checkbox(
        "Военный объект в городе",
        help="Наличие военных объектов в пределах города"
    )

    city_incorporated = st.checkbox(
        "Инкорпорированный город",
        help="Флаг официального административного статуса города"
    )

    st.subheader("🏫 Образование")

    schools_mean_distance = st.number_input(
        "Среднее расстояние до школ",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        help="Среднее расстояние от объекта до ближайших школ"
    )

    schools_min_distance = st.number_input(
        "Минимальное расстояние до школы",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        help="Минимальное расстояние от объекта до ближайшей школы"
    )

    schools_grouped = st.selectbox(
        "Количество школ поблизости",
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+", "20+"],
        help="Количество образовательных учреждений в окрестности"
    )

    st.subheader("📅 Экономические признаки")

    year_cat = st.selectbox(
        "Экономический период",
        ["Early period", "Primary period", "Growth period",
         "Peak construction period", "Upswing and subsequent crisis period",
         "Recovery and industry transformation period",
         "Decline and stabilization period"],
        help="Экономический и строительный период, в который был построен объект"
    )

    flag_status = st.checkbox(
        "Редкий статус объявления",
        help="Флаг, указывающий на редкость статуса объекта в данных"
    )

    flag_property = st.checkbox(
        "Редкий тип недвижимости",
        help="Флаг, указывающий на редкость типа недвижимости"
    )

    submitted = st.form_submit_button("🔮 Рассчитать стоимость")


# Предсказание + логирование
if submitted:
    input_df = pd.DataFrame([{
        "state": state,
        "group_status": group_status,
        "group_property": group_property,
        "remodeled_year": remodeled_year,
        "group_heating": group_heating,
        "group_cooling": group_cooling,
        "group_parking": group_parking,
        "lotsize_sqft": lotsize_sqft,
        "pool_bool": pool_bool,
        "fireplace_clear": fireplace_clear,
        "area": area,
        "schools_mean_distance": schools_mean_distance,
        "schools_min_distance": schools_min_distance,
        "city_category": city_category,
        "year_cat": year_cat,
        "flag_status": flag_status,
        "flag_property": flag_property,
        "baths": baths,
        "beds": beds,
        "stories_grouped": stories_grouped,
        "schools_grouped": schools_grouped,
        "city_military": city_military,
        "city_incorporated": city_incorporated,
        "from_center": from_center,
        "city_class": city_class,
        "density_class": density_class
    }])

    try:
        prediction = predict_new_data(input_df)

        log_row = input_df.copy()
        log_row["prediction"] = prediction
        log_row["timestamp"] = datetime.now()

        if os.path.exists(LOG_PATH):
            log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            log_row.to_csv(LOG_PATH, index=False)

        st.success(f"Прогнозируемая стоимость: ${prediction:,.0f}.")

    except Exception as e:
        st.error(f"Ошибка при расчёте: {e}")
