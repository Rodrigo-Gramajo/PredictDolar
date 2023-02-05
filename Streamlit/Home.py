import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import streamlit.components.v1 as components
import webbrowser


st.set_page_config(page_title="EcoPredictor")
image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\eco Predictor.png')

with st.sidebar.title("EcoPredictor"):
    with st.sidebar.title("EcoPredictor"):
        st.markdown('Aplicacion diseñada con Machine Learning para predecir variables Macro & Micro economicas de Argentina con la posibilidad de analizar graficos Historicos 😎')


st.image(image, caption=None, width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="PNG")

st.title("Bienvenidos a EcoPredictor")
st.write("This is a demo of Streamlit's capabilities for building amazing data apps.")

flujo_image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\Flujo EcoPredict.drawio.png')

st.title("Flujograma")
st.image(flujo_image, caption=None, width=None, use_column_width='auto', clamp=False, channels="RGB", output_format="PNG")

def download_file(url):
    webbrowser.open(url)

if st.button("Mostrar Flujograma"):
    url = r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\svg\Flujo EcoPredict.drawio.svg"
    download_file(url)

