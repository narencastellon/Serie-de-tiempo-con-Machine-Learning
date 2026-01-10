import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Configuración de la página para un look profesional
st.set_page_config(
    page_title="Forecasting de Demanda - Supplement Sales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS profesional (colores neutros, tipografía limpia, similar a dashboards empresariales)
st.markdown("""<style>
    .main { background-color: #f0f4f8; }  /* Fondo principal cambiado a gris claro suave para mejor visibilidad de textos */
    .stButton>button { background-color: #0056b3; color: white; border-radius: 5px; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #003f88; }
    .stSlider .stSliderLabel { color: #333333; font-family: Arial, sans-serif; }
    .stSelectbox { background-color: #ffffff; border: 1px solid #ced4da; border-radius: 5px; }
    h1, h2, h3 { color: #1a1a1a; font-family: Arial, sans-serif; }
    .sidebar .sidebar-content { background-color: #e9ecef; box-shadow: 2px 0 5px rgba(0,0,0,0.1); }  /* Fondo sidebar gris medio para contraste */
    .block-container { padding: 20px; background-color: "#6fab6c"; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    body { color: #333333; }  /* Asegurar texto oscuro para legibilidad */
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("📈 Forecasting de Demanda - Supplement Sales")
st.markdown("**App profesional para pronosticar demanda semanal por categoría usando modelo GPT-2 fine-tuned.**")

# Cargar dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/narencastellon/Serie-de-tiempo-con-Machine-Learning/refs/heads/main/Data/Supplement_Sales_Weekly_Expanded.csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Category', 'Date'])
    # Agrupar demanda semanal por Category
    df_category = df.groupby(['Category', 'Date'])['Units Sold'].sum().reset_index()
    df_category.rename(columns={'Units Sold': 'Demand'}, inplace=True)
    return df_category

df_category = load_data()

# Cargar modelo fine-tuned (asegúrate de tener la carpeta './gpt2-finetuned-demand' con el modelo guardado)
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned-demand')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned-demand')
    model.eval()  # Modo evaluación
    return model, tokenizer

model, tokenizer = load_model()

# Sidebar para navegación e inputs
st.sidebar.header("Configuración de Pronóstico")
categories = sorted(df_category['Category'].unique())
selected_category = st.sidebar.selectbox("Selecciona Categoría", categories)

horizon = st.sidebar.slider("Horizonte de Pronóstico (semanas)", 1, 12, 4)
context_length = st.sidebar.slider("Longitud del Contexto Histórico (semanas)", 10, 40, 20)

# Mostrar datos históricos de la categoría seleccionada
st.subheader(f"Histórico de Demanda - {selected_category}")
category_data = df_category[df_category['Category'] == selected_category]

if len(category_data) < context_length:
    st.warning(f"Datos insuficientes ({len(category_data)} semanas). Usando todo disponible.")
    context_length = len(category_data)

# Gráfico histórico
fig_hist = plt.figure(figsize=(12, 6))
sns.lineplot(data=category_data, x='Date', y='Demand', marker='o', color='#0056b3')
plt.title(f'Demanda Histórica Semanal - {selected_category}')
plt.xlabel('Fecha')
plt.ylabel('Demanda (Unidades Vendidas)')
plt.grid(True, alpha=0.3)
st.pyplot(fig_hist)

# Botón para generar pronóstico
if st.button("Generar Pronóstico", type="primary"):
    with st.spinner("Generando pronóstico con modelo GPT-2 fine-tuned..."):
        # Preparar contexto histórico
        historical = category_data['Demand'].values[-context_length:].tolist()
        input_text = f"Category: {selected_category} | Historical weekly demand: {', '.join(map(str, map(int, historical)))} | Next week demand:"
        
        forecast = []
        current_context = historical.copy()
        
        for _ in range(horizon):
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                predicted = int(new_text.split("Next week demand:")[-1].strip().split()[0])
            except:
                predicted = int(np.mean(current_context[-5:]))  # Fallback conservador
            forecast.append(max(predicted, 0))  # Asegurar no negativo
            current_context.append(predicted)
            # Actualizar prompt para próximo paso (autoregresivo)
            input_text = f"Category: {selected_category} | Historical weekly demand: {', '.join(map(str, map(int, current_context[-context_length:])))} | Next week demand:"
        
        # Crear fechas futuras
        last_date = category_data['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W')
        
        # Mostrar resultados
        st.success("¡Pronóstico generado exitosamente!")
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast_Demand': forecast})
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pronóstico Semanal")
            st.dataframe(forecast_df.style.format({'Forecast_Demand': '{:.0f}'}))

        with col2:
            st.subheader("Gráfico: Histórico + Pronóstico")
            combined = pd.concat([
                category_data[['Date', 'Demand']].rename(columns={'Demand': 'Value'}),
                pd.DataFrame({'Date': future_dates, 'Value': forecast})
            ])
            combined['Type'] = ['Histórico'] * len(category_data) + ['Pronóstico'] * horizon
            
            fig_forecast = plt.figure(figsize=(10, 6))
            sns.lineplot(data=combined, x='Date', y='Value', hue='Type', marker='o', palette=['#0056b3', '#e63946'])
            plt.title(f'Pronóstico de Demanda - {selected_category}')
            plt.xlabel('Fecha')
            plt.ylabel('Demanda (Unidades)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='')
            st.pyplot(fig_forecast)
        
        # Insights adicionales
        st.subheader("Insights")
        total_forecast = sum(forecast)
        avg_forecast = np.mean(forecast)
        st.markdown(f"""
        - **Demanda Pronosticada Total ({horizon} semanas):** {total_forecast:,.0f} unidades  
        - **Demanda Promedio Semanal:** {avg_forecast:,.0f} unidades  
        - **Recomendación:** Si la demanda pronosticada supera el stock actual, planificar reabastecimiento.
        """)

st.caption("App desarrollada con GPT-2 fine-tuned en datos de ventas de suplementos. Modelo experimental para demostración.")