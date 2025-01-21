# Manejo y Procesamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

import streamlit as st
from streamlit_option_menu import option_menu
# Cargando libreria para Modelos
from statsforecast import StatsForecast
from statsforecast.models import  AutoARIMA, SeasonalNaive, AutoETS

from univariado import show_univariado_page
from multivariado import show_multivariado_page
from multivariado import visual_multivariado
from univariado import visual_univariado

# Configuración básica para la aplicación
st.set_page_config(page_title = "Forecasting", layout="wide")

# Incluir el archivo CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")



# Configurar el sidebar
with st.sidebar:
    st.title("Navegación")
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Home", "Forecast", "Visualización"],
        icons=["house", "bar-chart-fill", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# Configurar las pestañas
if selected == "Home":
    st.title("Caso de Estudio: Adidas")
    st.write("¡Bienvenido a la página principal de la aplicación!")
    #st.write("Usa el menú en el lateral para navegar entre las diferentes secciones.")
    st.write(""" 
            La modelación de series de tiempo es una herramienta esencial en el análisis y pronóstico de ventas, y su aplicación al contexto del pronóstico de ventas de la marca ADIDAS resulta de gran relevancia. ADIDAS es una reconocida marca global de artículos deportivos, conocida por su amplia gama de productos y su presencia en múltiples mercados.

            El pronóstico preciso de las ventas de ADIDAS es fundamental para la planificación estratégica, la gestión de inventario, la toma de decisiones y el logro de los objetivos comerciales. La modelación de series de tiempo permite analizar y predecir los patrones y tendencias inherentes a los datos históricos de ventas, lo que ayuda a identificar los factores clave que influyen en el rendimiento de la marca y a tomar decisiones fundamentadas sobre la producción, distribución y estrategias de marketing.

            Al modelar las series de tiempo de las ventas de ADIDAS, se consideran diversos factores y variables, como la estacionalidad en la demanda de productos deportivos, las tendencias a largo plazo, los efectos promocionales, eventos deportivos importantes y otros factores económicos o sociales que pueden afectar las ventas de la marca. Estos factores se capturan mediante la aplicación de modelos de series de tiempo, como los modelos ARIMA, SARIMA, modelos de suavizamiento exponencial o modelos de redes neuronales, entre otros.

            La modelación de series de tiempo proporciona una visión profunda del comportamiento pasado de las ventas de ADIDAS y ayuda a predecir las ventas futuras de manera más precisa. Esto permite a la marca ADIDAS tomar decisiones estratégicas, optimizar la producción, planificar campañas de marketing y gestionar eficientemente el inventario en función de las demandas y patrones identificados.

            En resumen, la modelación de series de tiempo aplicada al pronóstico de ventas de la marca ADIDAS es una herramienta valiosa para comprender y predecir el rendimiento de la marca en función de los datos históricos. Esto ayuda a ADIDAS a tomar decisiones informadas y estratégicas, mejorando su capacidad para anticipar la demanda del mercado y mantener una ventaja competitiva en la industria de artículos deportivos.
        """)
    
elif selected == "Forecast":
    st.title("Forecast")
    
    forecast_tabs = st.tabs(["Pronóstico Univariado", "Pronóstico Multivariado"])
    
    with forecast_tabs[0]:
        st.header("Modelo Univariado")
        st.write("Esta es la sección de pronósticos Univariado.")
        st.write("  ")

        st.write(""" 
                Despues de entrenar el modelo y de realizar el forecasting vamos a presentar la 
                 evaluación de los modelos, usando diferente metricas
        """)
        st.write("  ")
    
        # Mostrar
        st.write(show_univariado_page())

    with forecast_tabs[1]:
        st.header("Modelo Multivariado")
        st.write("Esta es la sección de pronósticos Multivariado.")
        
        st.write("  ")

        st.write(""" 
                Despues de entrenar el modelo y de realizar el forecasting vamos a presentar la 
                 evaluación de los modelos, usando diferente metricas
        """)
        st.write("  ")
        # Mostrar
        st.write(show_multivariado_page())
    
elif selected == "Visualización":
    st.title("Visualización")
    st.write("Esta es la sección de visualización.")
    st.write("Aquí puedes añadir gráficos interactivos y análisis de datos.")


    visual_tab = st.tabs(["Visual Univariado", "Visual Multivariado"])
    
    with visual_tab[0]:
        st.header("Visual Univariado")
        st.write("Veamos el resultado del entrenamiento y forecasting del Modelo Univariado")
        
        # Mostrar
        st.write(visual_univariado())

    with visual_tab[1]:
        st.header("Visual Multivariado")
        st.write("Veamos el resultado del entrenamiento y forecasting del Modelo Multivariado")
        
        # Mostrar
        st.write(visual_multivariado())

# Firma en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### by: Naren Castellon")








