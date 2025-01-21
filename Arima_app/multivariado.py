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
 
# Cargando libreria para Modelos
from statsforecast import StatsForecast
from statsforecast.models import  AutoARIMA, SeasonalNaive, AutoETS

# Plot
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px




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
df_exo = df.groupby("ds")[["y",'Units Sold','Operating Profit','Total Cost']].sum().reset_index()
df_exo["unique_id"] = "1"

    # Exogenas
exogen = df_exo[["ds","unique_id", 'Units Sold','Operating Profit','Total Cost']]

    # Dividimos los datos
train_exo = df_exo[df_exo.ds<='2021-12-01'] 
test_exo = exogen[(exogen['ds'] > '2021-12-01')]


# modelando
season_length = 7 # weekly data 
horizon = len(test_exo) # number of predictions

    # We call the model that we are going to use
models_exo = [AutoARIMA(season_length=season_length),
                  AutoETS(season_length=season_length),
                  SeasonalNaive(season_length=season_length)]
    
    # Instantiate StatsForecast class as sf
sf_exo = StatsForecast(
        models = models_exo,
        freq = 'D', 
        n_jobs=-1)
    
    # Entrenando el modelo

sf_exo.fit(train_exo)

    # Forecast
Y_hat_exo = sf_exo.predict(h=horizon,  X_df=test_exo) 

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

actual_exo = df_exo[df_exo.ds>'2021-12-01'] 

# Lo que se muestra el Tab de Forecast
def show_multivariado_page(): 

    # Mostrar los Forecast
    st.subheader('Forecast')
    st.write(Y_hat_exo)
    
    # Se muestran las metricas
    st.write(evaluate_performace(train_exo, actual_exo,Y_hat_exo.reset_index() , models=["AutoARIMA","AutoETS", "SeasonalNaive"]))


def visual_multivariado():

   # Graficar los resultados con matplotlib 
    #st.write("Gráfico de Pronóstico") 
    fig, ax = plt.subplots(figsize = (18,6)) 
    ax.plot(df_exo['ds'][-80:], df_exo['y'][-80:], label='Valor Real') 
    ax.plot(Y_hat_exo['ds'], Y_hat_exo['AutoARIMA'], label = 'ARIMA', linestyle='--', color = "black", linewidth = 2) 
    ax.plot(Y_hat_exo['ds'], Y_hat_exo['AutoETS'], label = 'ETS', linestyle='--', color = "yellow", linewidth = 2) 
    ax.plot(Y_hat_exo['ds'], Y_hat_exo['SeasonalNaive'], label = 'SeasonalNaive', linestyle='--', color = "red", linewidth = 2) 
    ax.set_title("Gráfico Modelo Multivariado", fontsize = 24)
    ax.set_xlabel('Fecha') 
    ax.set_ylabel('Ventas($)') 
    ax.legend() 
    st.pyplot(fig)