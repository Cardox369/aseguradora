import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image

# --- Custom Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
    }
    .stApp {
        max-width: 800px !important;
        margin: 0 auto;
        padding: 2rem;
    }
    .st-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .logo-img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    .st-header h1 {
        color: #007bff;
        text-align: center;
    }
    .st-subheader {
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .st-selectbox label, .st-slider label, .st-number-input label {
        color: #555;
        font-weight: bold;
    }
    .st-selectbox div > div > div > div, .st-slider div > div > div, .st-number-input div > div > input {
        border-color: #ccc;
        border-radius: 5px;
    }
    .st-button > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
    }
    .st-button > button:hover {
        background-color: #0056b3;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        text-align: center;
    }
    .high-risk {
        color: #dc3545;
    }
    .low-risk {
        color: #28a745;
    }
    .model-info {
        margin-top: 1rem;
        font-style: italic;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load the Model ---
try:
    with open('modelo-clas-tree-knn-nn.pkl', 'rb') as file:
        model_Knn, model_Tree, model_NN, labelencoder, model_variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("El archivo del modelo 'modelo-clas-tree-knn-nn.pkl' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

# --- Main Content ---
st.container()

# --- Logo and Title ---
st.image("autos.jpg", caption="SafeDrive Risk Analyzer", width=150, use_container_width=True)
st.header("SafeDrive Risk Analyzer")
st.subheader("Predicción de Riesgo para Aseguradora")

# --- Parameters ---
age = st.slider("Seleccione la edad del vehículo:", min_value=18, max_value=100, value=33, step=1)
vehicle_type = st.selectbox("Seleccione el tipo de vehículo:", ["combi", "family", "sport", "minivan"])
selected_model_name = st.selectbox("Seleccione el modelo de predicción:", ["Nn", "Knn", "Dt"])

if st.button("Realizar Predicción"):
    # Crear DataFrame con los datos del usuario
    user_data = pd.DataFrame({'age': [age], 'cartype': [vehicle_type]}) # Cambiamos 'vehicle_type' a 'cartype'

    # Preprocesamiento (One-Hot Encoding para el tipo de vehículo)
    user_data = pd.get_dummies(user_data, columns=['cartype'], drop_first=False) # Usamos 'cartype' aquí

    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    if 'model_variables' in locals():
        processed_data = pd.DataFrame(columns=model_variables)
        for col in user_data.columns:
            if col in processed_data.columns:
                processed_data[col] = user_data[col]
        processed_data = processed_data.fillna(0) # Llenar las columnas faltantes con 0
    else:
        st.error("Las variables del modelo no se cargaron correctamente.")
        st.stop()

    # Seleccionar el modelo basado en la elección del usuario
    if selected_model_name == "Knn":
        selected_model = model_Knn
    elif selected_model_name == "Dt":
        selected_model = model_Tree
    elif selected_model_name == "Nn":
        selected_model = model_NN
    else:
        st.error("Modelo no reconocido")
        st.stop()

    try:
        # Realizar la predicción
        prediction = selected_model.predict(processed_data)

        # Mostrar la predicción
        st.subheader("Resultado de la Predicción:")
        if prediction[0] == 0:  # 0 represents high risk
            st.markdown(f"<p class='prediction-result high-risk'>Alto Riesgo</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='prediction-result low-risk'>Bajo Riesgo</p>", unsafe_allow_html=True)

        st.markdown(f"<p class='model-info'>Modelo utilizado: {selected_model_name}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")