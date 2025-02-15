# Manipulacion y tratamiento de datos
import pandas as pd
import numpy as np

# Desarrollo Apps 
#!pip install streamlit
import streamlit as st

# Modelacion
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

# Visualizacion - Plot
import plotly.express as px 
import plotly.graph_objects as go

df_sale = pd.read_csv("ventas.csv")

# Agregar una nueva variable
df_sale["total"] = df_sale["ventas"] * df_sale["precio_unitario"] * (1- df_sale["descuento"])

# Dividir los datos para el entrenamiento

X = df_sale[["precio_unitario", "descuento"]]
y = df_sale["ventas"]

# Dividir los datos en entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.20, shuffle= False)

# Entrenar los modelo
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestRegressor(n_estimators= 100)
model_rf.fit(X_train, y_train)

model_knn = KNeighborsRegressor(n_neighbors= 5)
model_knn.fit(X_train, y_train)

model_xgb = XGBRegressor(n_estimators = 100)
model_xgb.fit(X_train, y_train)

# Predicciones
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_knn = model_knn.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)

# Analisis de Caracterisitcas para Random Forest y XGBoost
importance_rf = model_rf.feature_importances_
importance_xgb = model_xgb.feature_importances_
features = X.columns


# Titulo de la aplicacion
st.title("Análisis de Ventas con Machine Learning")


# Mostrar datos
st.subheader("Datos de Ventas")
st.write(df_sale)

# Seleccionar el modelo
model_choice = st.sidebar.selectbox('Seleciona el Modelo de ML', ["Regresión Lineal", "Random Forest", "KNN","XGBoost"])

# Horizonte de pronostico
horizonte = st.sidebar.slider("Horizonte del Pronostico (mensual)", min_value = 1, max_value = 60, value = 12)

# generar datos sinteticos
future_dates = pd.date_range(start = df_sale.iloc[-12,0], periods = horizonte + 1, freq = "MS")[1:]
future_data = pd.DataFrame({
    "fecha":future_dates,
    "precio_unitario": np.random.uniform(5.0, 15.0 ,size = (horizonte,)),
    "descuento": np.random.uniform(0, 0.25, size = (horizonte,))
})

future_X = future_data[["precio_unitario", "descuento"]]

# Hacer predicciones

future_pred_lr = model_lr.predict(future_X)
future_pred_rf = model_rf.predict(future_X)
future_pred_knn = model_knn.predict(future_X)
future_pred_xgb = model_xgb.predict(future_X)

# Agregar las predicciones al DF
future_data["ventas_lr"] = future_pred_lr
future_data["ventas_rf"] = future_pred_rf
future_data["ventas_knn"] = future_pred_knn
future_data["ventas_xgb"] = future_pred_xgb

# Entrada del usuario

precio_unitario = st.sidebar.slider("Precio Unitario", min_value = 5.0, max_value = 20.0, step = 0.1)
descuento = st.sidebar.slider("Descuento", min_value = 0.0, max_value = 0.25, step = 0.01)

# prediccion
if st.button("Predecir Ventas"):
    if model_choice == "Regresion Lineal":
        ventas_pred = model_lr.predict([[precio_unitario, descuento]])[0]
    elif model_choice == "Random Forest":
        ventas_pred = model_rf.predict([[precio_unitario, descuento]])[0]
    elif model_choice == "KNN":
        ventas_pred = model_knn.predict([[precio_unitario, descuento]])[0]
    elif model_choice == "XGBoost":
        ventas_pred = model_xgb.predict([[precio_unitario, descuento]])[0]

    st.write(f'Prediccion de Ventas : {ventas_pred:.2f}')


# Mostrar pronostico o Forecast
st.subheader("Forecast")
st.write(future_data)

# Gráfico de tendencia de ventas con pronósticos
st.subheader('Tendencia de Ventas con Pronósticos')
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=df_sale['fecha'][-50:], y=df_sale['ventas'][-50:], mode='lines', name='Ventas Reales'))
if model_choice == 'Regresión Lineal':
    fig_trend.add_trace(go.Scatter(x=future_data['fecha'], y=future_data['ventas_lr'], mode='lines', name='Pronóstico LR', ))
elif model_choice == 'Random Forest':
    fig_trend.add_trace(go.Scatter(x=future_data['fecha'], y=future_data['ventas_rf'], mode='lines', name='Pronóstico RF', ))
elif model_choice == 'KNN':
    fig_trend.add_trace(go.Scatter(x=future_data['fecha'], y=future_data['ventas_knn'], mode='lines', name='Pronóstico KNN', ))
elif model_choice == 'XGBoost':
    fig_trend.add_trace(go.Scatter(x=future_data['fecha'], y=future_data['ventas_xgb'], mode='lines', name='Pronóstico XGBoost', ))
fig_trend.update_layout(title='Tendencia de Ventas con Pronósticos', xaxis_title='Fecha', yaxis_title='Ventas')
st.plotly_chart(fig_trend)

# Gráficos de análisis de características
st.subheader('Análisis de Características')
if model_choice == 'Random Forest':
    fig_rf = px.bar(x=features, y=importance_rf, labels={'x':'Característica', 'y':'Importancia'}, title='Importancia de Características - Random Forest')
    st.plotly_chart(fig_rf)
elif model_choice == 'XGBoost':
    fig_xgb = px.bar(x=features, y=importance_xgb, labels={'x':'Característica', 'y':'Importancia'}, title='Importancia de Características - XGBoost')
    st.plotly_chart(fig_xgb)
else:
    st.write("El análisis de características solo está disponible para Random Forest y XGBoost.")

# Firma 
st.sidebar.markdown("---")
st.sidebar.markdown("## By: Naren Castellon")