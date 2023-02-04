import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo



st.set_page_config(page_title="Prediccion Brecha entre Dolares")

with st.sidebar.title("Prediccion Brecha entre Dolares"):
    with st.sidebar.title("Prediccion Brecha entre Dolares"):
            st.markdown('Visualiza los datos historicos de las 6 variables que tienen mas del 80% de correlacion :sunglasses:')


header = st.container()
dataset = st.container()
features = st.container()
Brecha = st.container()

with header:
    st.title("Prediccion Brecha entre Dolares")

#Brecha.ipynb

def Brecha():

    import pandas as pd

    of = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newOf.xlsx')
    bl = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newBlue.xlsx')

    print(of)
    print(bl)


    merged_df = pd.merge(of, bl, on='Unnamed: 0')

    merged_df.to_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\p_merged_brecha.xls.pkl')

    print(merged_df)


    df = merged_df.rename(columns={'Unnamed: 0': 'Fecha', 'Values_x': 'Dolar Oficial', 'Values_y': 'Dolar Blue'})
 

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    def plot_dolar_chart(df):
        # calculate the gap in percentage
        df["gap"] = ((df["Dolar Blue"] - df["Dolar Oficial"]) / df["Dolar Oficial"]) * 100

        # Create the trace for Dolar Oficial
        trace1 = go.Scatter(x=df["Fecha"], y=df["Dolar Oficial"], mode='lines+markers', name='Dolar Oficial')

        # Create the trace for Dolar Blue
        trace2 = go.Scatter(x=df["Fecha"], y=df["Dolar Blue"], mode='lines+markers', name='Dolar Blue')

        # Create the trace for Gap
        trace3 = go.Scatter(x=df["Fecha"], y=df["gap"], mode='lines+markers', name='Brecha en porcentaje')

        # Create the data array
        data = [trace1, trace2, trace3]

        # Define the layout of the chart
        layout = go.Layout(title='Predicciones Dolar Oficial y Dolar Blue con su Brecha en porcentaje', xaxis=dict(title='Fecha'), yaxis=dict(title='Valores'))

        # Create the figure
        fig = go.Figure(data=data, layout=layout)
        
        return fig

    # Use the st.plotly_chart method to display the chart
    st.plotly_chart(plot_dolar_chart(df))



if st.button("Predecir"):
    Brecha()
