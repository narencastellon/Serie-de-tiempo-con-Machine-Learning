# =======================================================
# App Streamlit Dinámica: Pruebas de Stress en Forecasting de Ventas en Cadenas de Farmacias
# Usa NeuralForecast para predicciones con exógenas/futuras
# Incluye: Imagen header, menú lateral, datos históricos, EDA, predicciones dinámicas (selectbox farmacia, columnas para sliders, botón),
# sección interactiva de stress test (selectbox farmacia, sliders por escenario)
# Ejecuta con: streamlit run app_stress_forecasting_farmacias.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy matplotlib seaborn neuralforecast

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
import warnings
warnings.filterwarnings('ignore')

# Configuración página
st.set_page_config(page_title="Stress Forecasting Ventas Farmacias - Naren Castellón", layout="wide")
st.title("📈 Pruebas de Stress en Forecasting de Ventas en Cadenas de Farmacias")
st.markdown("**NeuralForecast para predicciones y escenarios de stress** | Dictado por Naren Castellón (@NarenCastellon)")

# Imagen header
st.sidebar.image("https://images.unsplash.com/photo-1587854692152-cbe660dbde88?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Gestión de ventas en farmacias con IA y stress testing",)



# =======================================================
# Carga de datos (cacheado)
# =======================================================
@st.cache_resource
def load_data():
    df = pd.read_csv('./Ventas_Cadenas_Farmacias.csv', parse_dates= ['ds'])
    stores = [f'Farmacia_{i+1}' for i in range(5)]

    df_ts = df[['store_id', 'ds', 'y', 'season_factor', 'promotion_active', 'inflation_rate', 'economic_index', 'unemployment_rate']].copy()
    df_ts = df_ts.rename(columns={'store_id': 'unique_id'})

    exog_cols = ['season_factor', 'promotion_active', 'inflation_rate', 'economic_index', 'unemployment_rate']

    train_df = df_ts[df_ts['ds'] < '2024-01-01']
    test_df = df_ts[df_ts['ds'] >= '2024-01-01']

    return df, train_df, test_df, exog_cols, stores

df, train_df, test_df, exog_cols, stores = load_data()

# =======================================================
# Carga de modelo (cacheado por separado para dinámica)
# =======================================================
@st.cache_resource
def get_nf(horizon):
    exog_cols = ['season_factor', 'promotion_active', 'inflation_rate', 'economic_index', 'unemployment_rate']
    models = [NHITS(h=horizon, input_size=24, futr_exog_list=exog_cols, loss=MAE(), max_steps=300, random_seed=42)]

    nf = NeuralForecast(models=models, freq='M')
    nf.fit(df=train_df)

    return nf

# =======================================================
# Base forecast (cacheado)
# =======================================================
@st.cache_resource
def get_base_forecast():
    futr_df_base = pd.DataFrame()
    for uid in stores:
        last_row = train_df[train_df['unique_id'] == uid].iloc[-1]
        future_dates = pd.date_range('2024-01-01', '2026-12-01', freq='M')
        futr_uid = pd.DataFrame({
            'unique_id': [uid] * len(future_dates),
            'ds': future_dates
        })
        for col in exog_cols:
            last_val = last_row[col]
            futr_uid[col] = last_val
        futr_df_base = pd.concat([futr_df_base, futr_uid])

    nf_base = get_nf(len(future_dates))
    forecast_base = nf_base.predict(futr_df=futr_df_base)

    return forecast_base

forecast_base = get_base_forecast()

# =======================================================
# Menú lateral
# =======================================================
st.sidebar.title("Navegación")
section = st.sidebar.radio("Secciones", 
                           ["Datos Históricos", "EDA", "Predicciones", "Stress Test"])

# =======================================================
# Secciones
# =======================================================
if section == "Datos Históricos":
    st.header("📊 Datos Históricos")
    st.write("Dataset completo con 18 variables (muestra)")
    st.dataframe(df.tail(50))

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    st.write("Tendencia Ventas Totales")
    df_agg = df.groupby('ds')['y'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_agg['ds'], df_agg['y'])
    st.pyplot(fig)

    st.write("Correlación Variables")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.write("Distribución Ventas por Farmacia")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='store_id', y='y', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Históricos por Farmacia")
    selected_store_eda = st.selectbox("Seleccionar Farmacia", stores)
    store_df = df[df['store_id'] == selected_store_eda]
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(store_df['ds'], store_df['y'])
    ax.set_title(f'Histórico de Ventas para {selected_store_eda}')
    st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones Dinámicas")
    st.write("Selecciona farmacia, horizonte, ajusta exógenas en 2 columnas, y genera predicciones")

    selected_store = st.selectbox("Seleccionar Farmacia", stores)
    horizon_selected = st.slider("Horizonte de Forecast (meses)", 1, 48, 12)

    # Get nf for selected horizon
    nf_dynamic = get_nf(horizon_selected)

    # Construir futr_df para TODAS farmacias
    futr_df_dynamic = pd.DataFrame()
    for uid in stores:
        last_row = train_df[train_df['unique_id'] == uid].iloc[-1]
        future_dates = pd.date_range('2024-01-01', periods=horizon_selected, freq='M')
        futr_uid = pd.DataFrame({
            'unique_id': [uid] * len(future_dates),
            'ds': future_dates
        })
        shock_dict = {}
        if uid == selected_store:
            col1, col2 = st.columns(2)
            with col1:
                for i in range(len(exog_cols)//2):
                    col = exog_cols[i]
                    last_val = last_row[col]
                    shock = st.slider(f"Shock {col} (multiplicador)", 0.5, 1.5, 1.0, key=f"{col}_{uid}")
                    shock_dict[col] = shock
            with col2:
                for i in range(len(exog_cols)//2, len(exog_cols)):
                    col = exog_cols[i]
                    last_val = last_row[col]
                    shock = st.slider(f"Shock {col} (multiplicador)", 0.5, 1.5, 1.0, key=f"{col}_{uid}")
                    shock_dict[col] = shock
        else:
            for col in exog_cols:
                shock_dict[col] = 1.0

        for col in exog_cols:
            last_val = last_row[col]
            futr_uid[col] = last_val * shock_dict[col]

        futr_df_dynamic = pd.concat([futr_df_dynamic, futr_uid])

    if st.button("Generar Predicciones"):
        forecast_dynamic = nf_dynamic.predict(futr_df=futr_df_dynamic)

        # Filtrar para selected_store
        forecast_store = forecast_dynamic[forecast_dynamic['unique_id'] == selected_store]

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(train_df[train_df['unique_id'] == selected_store]['ds'], train_df[train_df['unique_id'] == selected_store]['y'], label='Histórico')
        ax.plot(test_df[test_df['unique_id'] == selected_store]['ds'], test_df[test_df['unique_id'] == selected_store]['y'], label='Real Test', alpha=0.7)
        ax.plot(forecast_store['ds'], forecast_store['NHITS'], label='Forecast Dinámico')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Información Adicional: Predicciones para " + selected_store)
        mean_pred = forecast_store['NHITS'].mean()
        std_pred = forecast_store['NHITS'].std()
        st.metric("Predicción Media", f"{mean_pred:.2f}")
        st.metric("Desviación Estándar", f"{std_pred:.2f}")

        st.write("Forecast Completo Dinámico para " + selected_store)
        st.dataframe(forecast_store)

elif section == "Stress Test":
    st.header("⚠️ Stress Test Interactivo")
    st.write("Selecciona farmacia y ajusta variables para escenarios")

    selected_store_stress = st.selectbox("Seleccionar Farmacia", stores)

    scenario = st.selectbox("Seleccionar Escenario", ["Crisis Económica", "Cambio Demanda"])

    # Construir futr_df_stress para TODAS farmacias
    futr_df_stress = pd.DataFrame()
    for uid in stores:
        last_row = train_df[train_df['unique_id'] == uid].iloc[-1]
        future_dates = pd.date_range('2024-01-01', '2025-12-01', freq='M')
        futr_uid = pd.DataFrame({
            'unique_id': [uid] * len(future_dates),
            'ds': future_dates
        })
        shock_dict = {}
        if uid == selected_store_stress:
            for col in exog_cols:
                last_val = last_row[col]
                if scenario == "Crisis Económica" and col in ['unemployment_rate', 'inflation_rate']:
                    shock = st.slider(f"Shock {col} (multiplicador)", 1.0, 2.0, 1.2, key=f"{col}_{uid}_stress")
                elif scenario == "Crisis Económica" and col == 'economic_index':
                    shock = st.slider(f"Shock {col} (multiplicador)", 0.5, 1.0, 0.85, key=f"{col}_{uid}_stress")
                elif scenario == "Cambio Demanda" and col == 'promotion_active':
                    shock = st.slider(f"Shock {col} (0-1)", 0.0, 1.0, 1.0, key=f"{col}_{uid}_stress")
                elif scenario == "Cambio Demanda" and col == 'season_factor':
                    shock = st.slider(f"Shock {col} (multiplicador)", 1.0, 1.5, 1.1, key=f"{col}_{uid}_stress")
                else:
                    shock = 1.0
                shock_dict[col] = shock
        else:
            for col in exog_cols:
                shock_dict[col] = 1.0

        for col in exog_cols:
            last_val = last_row[col]
            futr_uid[col] = last_val * shock_dict[col]

        futr_df_stress = pd.concat([futr_df_stress, futr_uid])

    horizon_selected = st.slider("Horizonte de Forecast (meses)", 1, 24, 12)
    nf2 = get_nf(horizon_selected)
    forecast_stress = nf2.predict(futr_df=futr_df_stress)

    # Filtrar para selected_store_stress
    forecast_stress_store = forecast_stress[forecast_stress['unique_id'] == selected_store_stress]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train_df[train_df['unique_id'] == selected_store_stress]['ds'], train_df[train_df['unique_id'] == selected_store_stress]['y'], label='Histórico')
    ax.plot(forecast_base[forecast_base['unique_id'] == selected_store_stress]['ds'], forecast_base[forecast_base['unique_id'] == selected_store_stress]['NHITS'], label='Base Forecast')
    ax.plot(forecast_stress_store['ds'], forecast_stress_store['NHITS'], label=f'{scenario} Forecast')
    ax.legend()
    st.pyplot(fig)

    delta = (forecast_stress_store['NHITS'].mean() - forecast_base[forecast_base['unique_id'] == selected_store_stress]['NHITS'].mean()) / forecast_base[forecast_base['unique_id'] == selected_store_stress]['NHITS'].mean() * 100
    st.metric("Variación Ventas bajo Stress", f"{delta:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")