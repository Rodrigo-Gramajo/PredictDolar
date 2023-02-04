import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Predictor PBI")

with st.sidebar.title("Grafico"):
    with st.sidebar.title("Predictor"):
            st.markdown('Visualiza los datos historicos de las 6 variables que tienen mas del 80% de correlacion :sunglasses:')


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Grafico de Variables")

df = pd.read_csv(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\csv\PBI.csv')

columns = ['PBI per Capita', 'Produccion Agricola $', 'Suscripciones a Celulares', 'Consumo Cerveza Per Capita', 'Consumo Vino Per Capita', 'Expectativa de Vida', 'Consumo Recursos Fosiles Per Capita']

fig = px.line(df, x='Año', y=columns)

st.write(fig, width=800, height=400)