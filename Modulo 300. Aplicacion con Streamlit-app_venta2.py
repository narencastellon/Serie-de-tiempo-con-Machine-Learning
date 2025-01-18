#Cursos de Series de tiempo con Machine Learning

#Modulo 300. Aplicacion con Streamlit

#                    Elaborado por: Naren Castellon

# Cargamos librerias
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Generar datos de ventas ficticios
np.random.seed(42)
dates = pd.date_range(start='2021-01-01', periods=100, freq='M')
sales_data = {
    'fecha': dates,
    'ventas': np.random.randint(50, 500, size=(100,)),
    'precio_unitario': np.random.uniform(5.0, 15.0, size=(100,)),
    'descuento': np.random.uniform(0, 0.25, size=(100,))
}
df_sales = pd.DataFrame(sales_data)
df_sales['total'] = df_sales['ventas'] * df_sales['precio_unitario'] * (1 - df_sales['descuento'])

# Preparar los datos para el entrenamiento
X = df_sales[['precio_unitario', 'descuento']]
y = df_sales['ventas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_knn = mean_squared_error(y_test, y_pred_knn)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Mostrar las métricas de los modelos
#st.write("Linear Regression MSE:", mse_lr)
#st.write("Random Forest MSE:", mse_rf)
#st.write("KNN MSE:", mse_knn)
#st.write("XGBoost MSE:", mse_xgb)

# Análisis de características para Random Forest y XGBoost
importance_rf = model_rf.feature_importances_
importance_xgb = model_xgb.feature_importances_
features = X.columns

## ----------------------

# Título de la aplicación
st.title('Análisis de Ventas con Machine Learning')

# Mostrar los datos
st.subheader('Datos de Ventas')
st.write(df_sales.head())

# Selección del modelo
model_choice = st.sidebar.selectbox('Selecciona el Modelo de ML', ['Linear Regression', 'Random Forest', 'KNN', 'XGBoost'])

# Horizonte del pronóstico
horizonte = st.sidebar.slider('Horizonte del Pronóstico (meses)', min_value=1, max_value = 60, value = 12)

# Generar fechas futuras para el pronóstico
future_dates = pd.date_range(start=dates[-12], periods=horizonte + 1, freq='M')[1:]
future_data = pd.DataFrame({
    'fecha': future_dates,
    'precio_unitario': np.random.uniform(5.0, 15.0, size=(horizonte,)),
    'descuento': np.random.uniform(0, 0.25, size=(horizonte,))
})
future_X = future_data[['precio_unitario', 'descuento']]

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
    if model_choice == 'Linear Regression':
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
fig_trend.add_trace(go.Scatter(x=df_sales['fecha'], y=df_sales['ventas'], mode='lines', name='Ventas Reales'))
if model_choice == 'Linear Regression':
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
