import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="EcoPredictor")

with st.sidebar.title("Graficos Historicos"):
    with st.sidebar.title("Graficos Historicos"):
            st.markdown('Visualiza los datos historicos de las 6 variables que tienen mas del 80% de correlacion :sunglasses:')


header = st.container()
dataset = st.container()
features = st.container()
Graficos = st.container()

with header:
    st.title("Graficos Historicos")

def Graficos():

#Variables de PBI

    import pandas as pd
    
    df = pd.read_csv(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\csv\PBI.csv')

    columns = ['PBI per Capita', 'Produccion Agricola $', 'Suscripciones a Celulares', 'Consumo Cerveza Per Capita', 'Consumo Vino Per Capita', 'Expectativa de Vida', 'Consumo Recursos Fosiles Per Capita']

    fig = px.line(df, x='Año', y=columns)

    fig.update_layout(
        title='Variables del PBI por año',
        xaxis_title='Año',
        yaxis_title='Valores'
    )

    st.write(fig, width=800, height=400)

#Hitos

    import requests
    import pandas as pd

    TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDU1Mjk2ODYsInR5cGUiOiJleHRlcm5hbCIsInVzZXIiOiJyb2RyaWdveWdyYW1ham9AZ21haWwuY29tIn0.igINIa9yg5n49tsbOYqA74REk8upmBaHRpDp1oXlyTTz_IB6VQpZIEQ5Z9SYcnq6AMYUrkSM9dL5YoUZpPca5g"
    API_URL = "https://api.estadisticasbcra.com/milestones"

    # Agregar el token al header de autorización
    headers = {"Authorization": "BEARER " + TOKEN}

    # Realizar el request
    response = requests.get(API_URL, headers=headers)

    # Obtener el JSON de la respuesta
    milestone = response.json()

    milestone = pd.DataFrame(milestone)

    milestone = milestone[milestone['d'] >= '2010-01-01']


    import plotly.graph_objs as go
    import plotly.offline as pyo

    # create the trace
    trace = go.Scatter(x=milestone['d'], y=milestone['e'], mode='markers', name='Hitos')

    # create the layout
    layout = go.Layout(title='Hitos', xaxis=dict(title='Fecha'), yaxis=dict(title='Evento'))

    # create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

#Inflacion

    import requests

    TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDU1Mjk2ODYsInR5cGUiOiJleHRlcm5hbCIsInVzZXIiOiJyb2RyaWdveWdyYW1ham9AZ21haWwuY29tIn0.igINIa9yg5n49tsbOYqA74REk8upmBaHRpDp1oXlyTTz_IB6VQpZIEQ5Z9SYcnq6AMYUrkSM9dL5YoUZpPca5g"
    API_URL = "https://api.estadisticasbcra.com/inflacion_mensual_oficial"

    # Agregar el token al header de autorización
    headers = {"Authorization": "BEARER " + TOKEN}

    # Realizar el request
    response = requests.get(API_URL, headers=headers)

    # Obtener el JSON de la respuesta
    inflacion = response.json()

    inflacion = pd.DataFrame(inflacion)

    inflacion.to_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\inflacion.xlsx.pkl')

    inflacion = inflacion[inflacion['d'] >= '2010-01-01']

    import plotly.graph_objs as go
    import plotly.offline as pyo

    # create the trace
    trace = go.Scatter(x=inflacion['d'], y=inflacion['v'], mode='lines', name='Inflacion Mensual %')

    # create the layout
    layout = go.Layout(title='Inflacion Mensual %', xaxis=dict(title='Fecha'), yaxis=dict(title='Valor'))

    # create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

#Inflacion mensual & Hitos

    import plotly.graph_objs as go
    import plotly.offline as pyo

    # create the trace for inflacion
    trace1 = go.Scatter(x=inflacion['d'], y=inflacion['v'], mode='lines', name='Inflacion Mensual %', yaxis='y1')

    # create the trace for milestones
    trace2 = go.Scatter(x=milestone['d'], y=milestone['e'], mode='markers', name='Milestones', yaxis='y2')

    # create the layout
    layout = go.Layout(title='Inflacion Mensual & Hitos', xaxis=dict(title='Fecha'), yaxis=dict(title='Valor', side='left', showgrid=False, zeroline=False), yaxis2=dict(title='Event', side='right', overlaying='y', showgrid=False, zeroline=False))

    # create the figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

#Dolares 

    import pandas as pd

    pbl = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosBlue.xlsx', engine='openpyxl')
    pof = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx', engine='openpyxl')

    pbl = pbl[['Fecha', 'Venta']]
    pof = pof[['Fecha', 'Venta']]


    p_merged_df = pd.merge(pof, pbl, on='Fecha')

    p_merged_df.to_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\p_merged_brecha.xls.pkl')

    p_merged_df= p_merged_df.rename(columns={'Venta_x': 'Dolar Blue', 'Venta_y': 'Dolar Oficial'})

    p_merged_df["Fecha"] = pd.to_datetime(p_merged_df["Fecha"])


    # Add a new column for the percentage difference
    p_merged_df['Percentage Difference'] = (p_merged_df['Dolar Oficial'] - p_merged_df['Dolar Blue']) / p_merged_df['Dolar Blue'] * 100


    # create the trace for Dolar Blue
    trace1 = go.Scatter(x=p_merged_df['Fecha'], y=p_merged_df['Dolar Blue'], mode='lines+markers', name='Dolar Oficial', 
                        line=dict(width=0.5, color='#1f77b4'),
                        marker=dict(size=5))

    # create the trace for Dolar Oficial
    trace2 = go.Scatter(x=p_merged_df['Fecha'], y=p_merged_df['Dolar Oficial'], mode='lines+markers', name='Dolar Blue', 
                        line=dict(width=0.5, color='#ff7f0e'),
                        marker=dict(size=5))

    layout = go.Layout(title='Dolar Blue & Dolar Oficial', xaxis=dict(title='Fecha'), yaxis=dict(title='Valores'))


    # create the figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

#Economia

    milestone["valor"] = 200

    import plotly.graph_objs as go
    import plotly.offline as pyo

    # create the trace for Inflation
    trace1 = go.Scatter(x=inflacion['d'], y=inflacion['v'], mode='lines', name='Inflacion Mensual %')

    # create the trace for Dolar Blue
    trace2 = go.Scatter(x=p_merged_df['Fecha'], y=p_merged_df['Dolar Blue'], mode='lines+markers', name='Dolar Oficial', line=dict(width=1))

    # create the trace for Dolar Oficial
    trace3 = go.Scatter(x=p_merged_df['Fecha'], y=p_merged_df['Dolar Oficial'], mode='lines+markers', name='Dolar Blue', line=dict(width=0.5))

    # create the trace for Milestones
    trace4 = go.Scatter(x=milestone['d'], y=milestone['valor'], mode='markers', name='Hito', text=milestone['e'])

    # create the layout
    layout = go.Layout(title='Inflacion, Dolar Blue, Dolar Oficial & Hitos', xaxis=dict(title='Fecha'))

    # create the figure
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

fig = Graficos()

