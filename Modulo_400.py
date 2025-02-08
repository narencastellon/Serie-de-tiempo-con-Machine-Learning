# Cursos de Series de tiempo con Machine Learning
# Modulo 400. Aplicación con Streamlit ML Forecasting Time Series

#                       Elaborado por: Naren Castellon

# Cargar las librerias 

# Manipulacion y tratamiento de Datos
import numpy as np
import pandas as pd

# Desarrollo de Apps
import streamlit as st

# Modelacion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Evaluar el modelo
from sklearn.metrics import mean_squared_error

# Visualizacion de datos
import plotly.express as px
import plotly.graph_objects as go


# Cargar los datos

df_sales = pd.read_csv("ventas.csv")

# agragar una nueva variable

df_sales["total"] = df_sales['ventas'] * df_sales['precio_unitario']*(1 - df_sales['descuento'])

lags = [1, 2,3]
# Crear las columnas de rezagos y añadirlas al DataFrame
for lag in lags:
    df_sales[f'lag_{lag}'] = df_sales['ventas'].shift(lag)


# Lista de ventanas de promedio móvil que quieres crear
windows = [2, 3, 4]

# Crear las columnas de promedio móvil y añadirlas al DataFrame
for window in windows:
    df_sales[f'SMA{window}'] = df_sales['ventas'].rolling(window=window).mean()



# Mostrar el DataFrame resultante
df_sales.dropna(inplace = True)

print(df_sales)

# Prepar los datos para el entrenamiento

X = df_sales[['precio_unitario', 'descuento', 'lag_1','lag_2','lag_3', 'SMA2', 'SMA3', 'SMA4' ]] # 'lag_1','lag_2', 'SMA2', 'SMA3', 'SMA4'
y = df_sales['ventas']

# Dividir los datos en entrenamiento y prueba

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size= 0.20, random_state= 42, shuffle= False)

# Entrenar modelos
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)

model_xgb = XGBRegressor(n_estimators=100, random_state=42)
model_xgb.fit(X_train, y_train)

# Predicciones y evaluación
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_knn = model_knn.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)

# Evaluar el modelo
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)

#print("Modelo lineal", mse_lr)
# print("Modelo Ensemble", mse_rf)

# Análisis de características para Random Forest y XGBoost
importance_rf = model_rf.feature_importances_
importance_xgb = model_xgb.feature_importances_
features = X.columns

# Título de la aplicación
st.title('Análisis de Ventas con Machine Learning')

# Mostrar los datos
st.subheader('Datos de Ventas')
st.write(df_sales)

# Selección del modelo
model_choice = st.sidebar.selectbox('Selecciona el Modelo de ML', ['Regresion Lineal', 'Random Forest', 'KNN', 'XGBoost'])

# Horizonte del pronóstico
horizonte = st.sidebar.slider('Horizonte del Pronóstico (meses)', min_value=1, max_value = 60, value = 12)

# Generar fechas futuras para el pronóstico
future_dates = pd.date_range(start= df_sales.iloc[-12,0], periods=horizonte + 1, freq='MS')[1:]
future_data = pd.DataFrame({
    'fecha': future_dates,
    'precio_unitario': np.random.uniform(5.0, 15.0, size=(horizonte,)),
    'descuento': np.random.uniform(0, 0.25, size=(horizonte,)),
    'lag_1': np.random.uniform(0, 600, size=(horizonte,)),
    'lag_2': np.random.uniform(0, 600, size=(horizonte,)),
    'lag_3': np.random.uniform(0, 600, size=(horizonte,)),
    'SMA2': np.random.uniform(0, 600, size=(horizonte,)),
    'SMA3': np.random.uniform(0, 600, size=(horizonte,)),
    'SMA4': np.random.uniform(0, 600, size=(horizonte,)),
})
future_X = future_data[['precio_unitario', 'descuento', 'lag_1','lag_2','lag_3', 'SMA2', 'SMA3', 'SMA4']] # 'lag_1','lag_2', 'SMA2', 'SMA3', 'SMA4'

# Hacer predicciones para el futuro
future_pred_lr = model_lr.predict(future_X)
future_pred_rf = model_rf.predict(future_X)
future_pred_knn = model_knn.predict(future_X)
future_pred_xgb = model_xgb.predict(future_X)

# Agregar las predicciones al DataFrame
future_data['ventas_lr'] = future_pred_lr
future_data['ventas_rf'] = future_pred_rf
future_data['ventas_knn'] = future_pred_knn
future_data['ventas_xgb'] = future_pred_xgb

# Entradas del usuario
precio_unitario = st.sidebar.slider('Precio Unitario', min_value=5.0, max_value=15.0, step=0.1)
descuento = st.sidebar.slider('Descuento', min_value=0.0, max_value=0.25, step=0.01)

# Predicción
if st.button('Predecir Ventas'):
    if model_choice == 'Regresion Lineal':
        ventas_pred = model_lr.predict([[precio_unitario, descuento]])[0]
    elif model_choice == 'Random Forest':
        ventas_pred = model_rf.predict([[precio_unitario, descuento]])[0]
    elif model_choice == 'KNN':
        ventas_pred = model_knn.predict([[precio_unitario, descuento]])[0]
    elif model_choice == 'XGBoost':
        ventas_pred = model_xgb.predict([[precio_unitario, descuento]])[0]
    
    st.write(f'Predicción de Ventas: {ventas_pred:.2f}')

# Mostrar los Forecast
st.subheader('Forecast')
st.write(future_data)

# Gráfico de tendencia de ventas con pronósticos
st.subheader('Tendencia de Ventas con Pronósticos')
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=df_sales['fecha'][-50:], y=df_sales['ventas'][-50:], mode='lines', name='Ventas Reales'))
if model_choice == 'Regresion Lineal':
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