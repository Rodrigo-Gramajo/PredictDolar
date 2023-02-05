import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="My Streamlit App", page_icon=":guardsman:", layout="wide")

with st.sidebar.title("Prediccion PBI per Capita"):
    with st.sidebar.title("Prediccion PBI per Capita"):
            st.markdown('Realiza predicciones de PBI per Capita usando 6 variables que tienen mas del 80% de correacion :sunglasses:')

st.title("Bienvenidos a EcoPredictor")

st.write("This is a demo of Streamlit's capabilities for building amazing data apps.")

# Add a button
if st.button("Explore the features"):
    st.write("This button takes you to the features page.")
