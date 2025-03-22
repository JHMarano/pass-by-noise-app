
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predição de Pass-by Noise", layout="centered")

st.title("🔊 Predição de Nível de Ruído Pass-by Noise [dB(A)]")

# Carrega o modelo treinado
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_pass_by_noise_colab.pkl")

modelo = carregar_modelo()

# Interface do usuário para entrada dos dados
st.subheader("📋 Insira os dados do veículo e do teste:")

with st.form("form_entrada"):
    col1, col2 = st.columns(2)

    with col1:
        vehicle_category = st.selectbox("Categoria do Veículo", ["M1", "N1", "N2"])
        vehicle_mass_kg = st.number_input("Massa do Veículo (kg)", 800, 8000, step=50, value=1600)
        pmr = st.number_input("PMR (Potência/Massa) [kW/t]", 10.0, 150.0, step=1.0, value=65.0)
        engine_type = st.selectbox("Tipo de Motor", ["gasoline", "diesel", "electric", "hybrid"])
        transmission_type = st.selectbox("Transmissão", ["manual", "automatic", "CVT"])
        number_of_gears = st.slider("Número de Marchas", 4, 10, value=5)
        tyre_type = st.selectbox("Tipo de Pneu", ["urban", "off-road", "low-profile"])

    with col2:
        gear_used = st.selectbox("Marcha Usada no Teste", [2, 3, 4, 5])
        speed_kph = st.number_input("Velocidade (km/h)", 30.0, 80.0, step=0.5, value=50.0)
        vehicle_position_m = st.number_input("Posição do Veículo na Pista (m)", -50.0, 50.0, step=0.1, value=0.0)
        distance_to_microphone_m = st.number_input("Distância até Microfone (m)", 6.0, 9.0, step=0.1, value=7.5)
        fuel_type = st.selectbox("Tipo de Combustível", ["gasoline", "diesel", "ethanol", "electric"])
        driving_mode = st.selectbox("Modo de Condução", ["WOT", "CRS"])
        pavement_type = st.selectbox("Tipo de Pavimento", ["ISO_10844", "asphalt", "concrete"])
        ambient_temp_C = st.number_input("Temperatura Ambiente (°C)", -10.0, 50.0, step=0.5, value=25.0)
        wind_speed_kph = st.number_input("Velocidade do Vento (km/h)", 0.0, 50.0, step=0.5, value=2.0)

    submitted = st.form_submit_button("🔍 Prever Ruído")

# Predição
if submitted:
    entrada = pd.DataFrame([{
        "vehicle_category": vehicle_category,
        "vehicle_mass_kg": vehicle_mass_kg,
        "pmr": pmr,
        "engine_type": engine_type,
        "transmission_type": transmission_type,
        "number_of_gears": number_of_gears,
        "tyre_type": tyre_type,
        "gear_used": gear_used,
        "speed_kph": speed_kph,
        "vehicle_position_m": vehicle_position_m,
        "distance_to_microphone_m": distance_to_microphone_m,
        "fuel_type": fuel_type,
        "driving_mode": driving_mode,
        "pavement_type": pavement_type,
        "ambient_temp_C": ambient_temp_C,
        "wind_speed_kph": wind_speed_kph
    }])

    pred = modelo.predict(entrada)[0]
    st.success(f"🎯 Previsão de Nível de Ruído: **{pred:.2f} dB(A)**")
