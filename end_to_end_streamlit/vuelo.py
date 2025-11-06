import streamlit as st

#Cargar modelo
import pickle

# manipulacion -tratamiento -tiempo de datos
import pandas as pd
import numpy as np
from datetime import date, time, datetime, timedelta

# --- 1. Cargar el modelo

try:
    #cargar el modelo Random Forest
    model = pickle.load(open("c2_flight_rf.pkl", "rb"))
except FileNotFoundError:
    st.error('Erro: El Archivo del modelo "c2_flight_rf.pkl" no se encuentra ')
    st.info("Por favor, asegurate de colocar el archivo del modelo en el mismo directorio que este script")
    model = None

# -- configuracion de la pagina

st.set_page_config(
    page_title = "Predicción de Precios de Vuelos",
    layout = "centered",
    initial_sidebar_state = "auto"
)

# Titulo de la aplicacion

st.title("✈️ Predictor de Precio de Vuelos")
st.markdown("Ingrese los detalles del vuelo para obtener una estimacion del precio")

# -- 2. Entradas del Usuario

st.header("1. Información de Tiempo")

col1, col2 = st.columns(2)

with col1:
    # Fecha de viaje (solo la fecha para dia y mes)
    journey_date_input = st.date_input(
        "Fecha de Partida (Dia/Mes del Viaje)",
        date.today(),
        min_value = date.today()
    )

    # Hora de salida
    dep_time_input = st.time_input("Hora de Salidad (HH:MM)", time(10, 0))

with col2:
    # Hora de llegada
    arr_time_input = st.time_input("Hora Estimada de Llegada (HH:MM)", time(12, 0))

    # Numero de escala
    total_stops = st.number_input(
        "Número de Escala (Total Stops)",
        min_value = 0,
        max_value =  4,
        step = 1,
        value = 1
    )

# --3. Areolinea, origen y destino

st.header("2. Aerolínea y Rutas")

col3, col4, col5 = st.columns(3)

Airlines = ['Jet Airways', 'IndiGo', 'Air India','Multiple carriers', 'SpiceJet', 'Vistara', 
            'GoAir', 'Other']

Source = ['Delhi', 'Kolkata','Mumbai', 'Chennai','Banglore'  ]

Destinations = ['Cochin','Delhi','Hyderabad', 'Kolkata', 'New Delhi', 'Banglore', 'Chennai']

with col3:
    airline = st.selectbox("Aerolina", Airlines)

with col4:
    source = st.selectbox("Ciudad de Origen (Source)", Source)

with col5:
    destination = st.selectbox("Ciudad de Destino (Destination)", Destinations)


# Logica de prediccion (Funcion principal)

if st.button("Calcular Precio del vuelo"):
    if model is None:
        st.warning("No se puede predecir sin el modelo cargado. Revise el archivo 'c2_flight_rf.pkl' ")
    else:
        with st.spinner('Calculando predicción ...'):

            # 1. Extraer y conversion de fechas/ tiempos

            # Journey Day & Month

            journey_day = journey_date_input.day
            journey_month = journey_date_input.month

            #departure time

            dep_hour = dep_time_input.hour
            dep_min = dep_time_input.minute

            # Arrival time

            arrival_hour = arr_time_input.hour
            arrival_min = arr_time_input.minute

            # Duracion

            Duration_hour = abs(arrival_hour - dep_hour)
            Duration_mins = abs(arrival_min - dep_min)

            # Total stops
            Total_Stops = int(total_stops)

            # 2. Codificar 

            Airline_AirIndia = 0
            Airline_GoAir = 0
            Airline_IndiGo = 0
            Airline_JetAirways = 0
            Airline_MultipleCarriers = 0
            Airline_SpiceJet = 0
            Airline_Vistara = 0
            Airline_Other = 0

             # Asignar 1 a la aerolínea seleccionada
            if airline == 'Jet Airways':
                Airline_JetAirways = 1
            elif airline == 'IndiGo':
                Airline_IndiGo = 1
            elif airline == 'Air India':
                Airline_AirIndia = 1
            elif airline == 'Multiple carriers':
                Airline_MultipleCarriers = 1
            elif airline == 'SpiceJet':
                Airline_SpiceJet = 1
            elif airline == 'Vistara':
                Airline_Vistara = 1
            elif airline == 'GoAir':
                Airline_GoAir = 1
            else:
                Airline_Other = 1

            
            # 3. ONE-HOT ENCODING para Origen (Source)
            # Inicializar todas las variables de Origen a 0
            Source_Chennai = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
            # Bangalore no está en la lista de variables OHE, y Delhi se infiere si no es ninguna de las anteriores en el Flask original.
            # Replicamos el comportamiento exacto del Flask original:
            if source == 'Kolkata':
                Source_Kolkata = 1
            elif source == 'Mumbai':
                Source_Mumbai = 1
            elif source == 'Chennai':
                Source_Chennai = 1
            # Si es 'Delhi' o 'Bangalore' o 'Other', las variables se mantienen en 0.

            # 4. ONE-HOT ENCODING para Destino (Destination)
            # Inicializar todas las variables de Destino a 0
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            # New Delhi, Bangalore, y Chennai se infieren o no se usan en el OHE
            # Replicamos el comportamiento exacto del Flask original:
            if destination == 'Cochin':
                Destination_Cochin = 1
            elif destination == 'Delhi':
                Destination_Delhi = 1
            elif destination == 'Hyderabad':
                Destination_Hyderabad = 1
            elif destination == 'Kolkata':
                Destination_Kolkata = 1
            # Si es 'New Delhi', 'Bangalore', o 'Chennai' las variables se mantienen en 0.

            # 5. Creación del vector de características para la predicción
            # El orden de las características debe COINCIDIR EXACTAMENTE con el orden del entrenamiento del modelo.
            features = [
                Total_Stops,
                journey_day,
                journey_month,
                dep_hour,
                dep_min,
                arrival_hour,
                arrival_min,
                Duration_hour,
                Duration_mins,
                # Airlines OHE
                Airline_AirIndia,
                Airline_GoAir,
                Airline_IndiGo,
                Airline_JetAirways,
                Airline_MultipleCarriers,
                Airline_Other, # El Flask original tenía Airline_Other después de MultipleCarriers
                Airline_SpiceJet,
                Airline_Vistara,
                # Sources OHE
                Source_Chennai,
                Source_Kolkata,
                Source_Mumbai,
                # Destinations OHE
                Destination_Cochin,
                Destination_Delhi,
                Destination_Hyderabad,
                Destination_Kolkata,
            ]

            # Convertir a array de numpy y redimensionar para la predicción
            final_features = np.array([features])

            # 6. prediccion
            prediction = model.predict(final_features)
            output = round(prediction[0], 2)

            # 7. Mostrar el resultado
            st.success(f"La predicción de Precio para su Vuelo es : **${output:,.2f}**")
            st.balloons()

            