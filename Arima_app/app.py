import streamlit as st
from streamlit_option_menu import option_menu

# Configuración básica para la aplicación
st.set_page_config(page_title = "Forecast", layout="wide")

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
    st.title("Home")
    st.write("¡Bienvenido a la página principal de la aplicación!")
    st.write("Usa el menú en el lateral para navegar entre las diferentes secciones.")
    st.header("En estamos construyendo esta aplicacion para un modelo de Machine Learning para nuestro proximo video que vamos hacer en nuestro canal de youtube")
    st.header("y para nuestro curso de Especializacion en Forecast.")
    
elif selected == "Forecast":
    st.title("Forecast")
    st.write("Esta es la sección de pronósticos.")
    st.write("Aquí puedes añadir tu lógica de predicción y gráficos relacionados.")
    
elif selected == "Visualización":
    st.title("Visualización")
    st.write("Esta es la sección de visualización.")
    st.write("Aquí puedes añadir gráficos interactivos y análisis de datos.")
    
# Un mensaje de bienvenida en el sidebar
st.sidebar.info("Selecciona una pestaña para comenzar.")

# Firma en el sidebar 
st.sidebar.markdown("---") 
st.sidebar.markdown("### by: Naren Castellon")
