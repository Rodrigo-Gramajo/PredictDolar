import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="EcoPredictor")

with st.sidebar.title("Prediccion PBI per Capita"):
    with st.sidebar.title("Prediccion PBI per Capita"):
        st.markdown('Realiza predicciones de PBI per Capita usando 6 variables que tienen mas del 80% de correacion ⚖️')            with st.sidebar.container():
             image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\eco Predictor.png')
             st.image(image, use_column_width=True)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Prediccion PBI per Capita")

with features:
    st.markdown('6 Variables con mas correlacion al PBI per Capita*')
    st.subheader("Produccion Agricola $")
    Produccion_Agricola0 = st.slider("Valor bruto de la producción del sector agropecuario, medido en US$ MM", 15, 65)
    Produccion_Agricola = format(Produccion_Agricola0, "01d")
    st.subheader("Suscripciones a Celulares")
    Suscripciones_a_Celulares0 = st.slider("Suscripciones a un servicio público de telefonía móvil en M", 60, 70)
    Suscripciones_a_Celulares = format(Suscripciones_a_Celulares0, "01d")
    st.subheader("Consumo Cerveza per Capita")
    Consumo_Cerveza_Per_Capita = st.slider("Consumo medio anual de cerveza per cápita, medido en litros de alcohol puro al año", 0, 5)
    st.subheader("Consumo Vino per Capita")
    Consumo_Vino_Per_Capita = st.slider("Consumo medio per cápita de vino, medido en litros de alcohol puro al año", 0, 5)
    st.subheader("Expectativa de Vida")
    Expectativa_de_Vida = st.slider("El número promedio de años que un recién nacido podría esperar vivir", 70, 80)
    st.subheader("Consumo Recursos Fosiles Per Capita")
    Consumo_Recursos_Fosiles_Per_Capita = st.slider("El consumo medio de energía procedente de carbón, petróleo y gas por persona en kWh", 10, 20)


with model_training:

    df = pd.read_csv(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\csv\PBI.csv')
    features = ['Produccion Agricola $', 'Suscripciones a Celulares', 'Consumo Cerveza Per Capita', 'Consumo Vino Per Capita', 'Expectativa de Vida', 'Consumo Recursos Fosiles Per Capita']
    X = df[features]
    y = df['PBI per Capita']

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    
    if st.button("Predecir"):
        input_values = [Produccion_Agricola, Suscripciones_a_Celulares, Consumo_Cerveza_Per_Capita, Consumo_Vino_Per_Capita, Expectativa_de_Vida, Consumo_Recursos_Fosiles_Per_Capita]
        prediction = model.predict([input_values])
        prediction_value = prediction[0] * 3.6
        formatted_prediction = "${:,.2f}".format(prediction_value)
        st.header("PBI per Capita: ")
        st.markdown("<span style='color: green; font-size: 24px;'>{}</span>".format(formatted_prediction), unsafe_allow_html=True)
