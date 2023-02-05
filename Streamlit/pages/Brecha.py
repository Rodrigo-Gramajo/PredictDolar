import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from PIL import Image

st.set_page_config(page_title="EcoPredictor")

with st.sidebar.title("Prediccion Brecha entre Dolares"):
    with st.sidebar.title("Prediccion Brecha entre Dolares"):
            st.markdown('Visualiza las prediciones de Dolar Oficial & Dolar Blue con su % de Brecha cambiaria 🤑')
            with st.sidebar.container():
             image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\eco Predictor.png')
             st.image(image, use_column_width=True)

header = st.container()
dataset = st.container()
features = st.container()
Brecha = st.container()

with header:
    st.title("Prediccion Brecha entre Dolares")

#Brecha.ipynb

def Brecha():

    of = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newOf.xlsx')
    bl = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newBlue.xlsx')

    merged_df = pd.merge(of, bl, on='Unnamed: 0')

    df = merged_df.rename(columns={'Unnamed: 0': 'Fecha', 'Values_x': 'Dolar Oficial', 'Values_y': 'Dolar Blue'})

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df["gap"] = ((df["Dolar Blue"] - df["Dolar Oficial"]) / df["Dolar Oficial"]) * 100

    fig = px.line(df, x="Fecha", y=["Dolar Oficial", "Dolar Blue", "gap"], title='Predicciones Dolar Oficial y Dolar Blue con su Brecha en porcentaje')

    return fig

fig = Brecha()

st.plotly_chart(fig)

if st.button("Actualizar"):
    fig = Brecha()