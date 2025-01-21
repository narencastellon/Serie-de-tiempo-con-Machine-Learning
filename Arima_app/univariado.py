# Manejo y Procesamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Apps
# ==============================================================================
import streamlit as st

# StatsForecast
# ==============================================================================
from statsforecast import StatsForecast
from utilsforecast.plotting import plot_series

# Plot
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px




# Función para centrar texto con HTML y CSS 
def centered_text(text):
    return f'<div class="centered-text">{text}</div>'



# Cargando los datos

df = pd.read_csv("https://raw.githubusercontent.com/Naren8520/Serie-de-tiempo-con-Machine-Learning/main/Data/Adidas%20US%20Sales%20Data.csv",sep=";")

    # We eliminate the dollar sign and the space from the comma
df['Price per Unit'] = df['Price per Unit'].str.replace('$', '')
df['Total Sales'] = df['Total Sales'].str.replace('$', '').str.replace(',', '')
df['Operating Profit'] = df['Operating Profit'].str.replace('$', '').str.replace(',', '')
df["Units Sold"]=df["Units Sold"].str.replace(',', '')

    # Remove the % sign and divide by 100
df['Operating Margin']=df['Operating Margin'].str[:-1].astype(float)
df['Operating Margin'] = df['Operating Margin'] / 100

    #Changing datatype of Invoice Date to datetime
df['Invoice Date']=pd.to_datetime(df['Invoice Date']) 

df[['Price per Unit', 'Units Sold', 'Total Sales','Operating Profit']] = df[['Price per Unit', 'Units Sold', 'Total Sales','Operating Profit']].astype("float")

df["unique_id"]="1"
df = df.rename(columns={"Invoice Date": "ds", "Total Sales": "y"})

df['Total Cost'] = df['y'] - df['Operating Profit']
df['Product Cost'] = df['Price per Unit'] - (df['Price per Unit'] * df['Operating Margin'])
df['Year'] = pd.to_datetime(df['ds']).dt.year
df['Month'] = pd.to_datetime(df['ds']).dt.month
df['Day'] = pd.to_datetime(df['ds']).dt.day

df['City_State'] = df['City'] + ', ' + df['State'] #Considering city names alone does not make sense, as some states have common city names.

    # agrupando
data = df.groupby("ds")[["y"]].sum().reset_index()
data["unique_id"]="1"

    # Division de Datos
train = data[data.ds<='2021-12-01'] 
test=data[(data['ds'] > '2021-12-01')]

    # Cargando libreria para Modelos
from statsforecast import StatsForecast
from statsforecast.models import  AutoARIMA, SeasonalNaive, AutoETS

season_length = 7 # weekly data 
horizon = len(test) # number of predictions

    # We call the model that we are going to use
models = [AutoARIMA(season_length=season_length),
          AutoETS(season_length=season_length),
          SeasonalNaive(season_length=season_length)]

    # Instantiate StatsForecast class as sf
sf = StatsForecast(
        df = train,
        models = models,
        freq ='D', 
        n_jobs = -1)

    # Entrenar el modelo
sf.fit()

    # Forecast
Y_hat = sf.predict(horizon)

# Evaluar el Modelo

from datasetsforecast.losses import (mae, mape, mase, rmse, smape)
from utilsforecast.evaluation import evaluate

def evaluate_performace(y_hist, y_true, y_pred, models):
    y_true = pd.merge(y_true,y_pred, how='left', on=['ds'])
    evaluation = {}
    for model in models:
        evaluation[model] = {}
        for metric in [mase, mae, mape, rmse, smape]:
            metric_name = metric.__name__
            if metric_name == 'mase':
                evaluation[model][metric_name] = metric(y_true['y'].values,
                                                 y_true[model].values,
                                                 y_hist['y'].values, seasonality=24)
            else:
                evaluation[model][metric_name] = metric(y_true['y'].values, y_true[model].values)
    return pd.DataFrame(evaluation).T


def show_univariado_page(): 
    
    col1, col2 = st.columns(2)
    with col1:
        # Mostrar los Forecast
        #st.subheader('Forecast')
        st.markdown(centered_text("Forecast"), unsafe_allow_html=True)
        st.write(Y_hat)
    with col2:
        # Mostrar las metricas del modelo
        #st.subheader('Metricas')
        #st.write(" Métricas del Modelo")
        st.markdown(centered_text(" Métricas del Modelo"), unsafe_allow_html=True)
        st.write(evaluate_performace(train, test,Y_hat.reset_index() , models=["AutoARIMA","AutoETS", "SeasonalNaive"]))

plt.style.use("classic") # fivethirtyeight  grayscale  classic
dark_style = {
    'axes.facecolor':"#98daa7"}  # '#484366'  '#008080' "#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af" 
plt.rcParams.update(dark_style)

def visual_univariado():

   # Graficar los resultados con matplotlib 
    #st.write("Gráfico de Pronóstico") 
    fig, ax = plt.subplots(figsize = (18,6)) 
    ax.plot(data['ds'][-80:], data['y'][-80:], label='Valor Real') 
    ax.plot(Y_hat['ds'], Y_hat['AutoARIMA'], label = 'ARIMA', linestyle='--', color = "red", linewidth = 2) 
    ax.plot(Y_hat['ds'], Y_hat['AutoETS'], label = 'ETS', linestyle='--', color = "black", linewidth = 2) 
    ax.plot(Y_hat['ds'], Y_hat['SeasonalNaive'], label = 'SeasonalNaive', linestyle='--', color = "fuchsia", linewidth = 2) 
    ax.set_title("Gráfico Modelos Univariados", fontsize = 24)
    ax.set_xlabel('Fecha') 
    ax.set_ylabel('Ventas($)') 
    ax.grid()
    ax.legend() 
    st.pyplot(fig)