import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import time
from PIL import Image


st.set_page_config(page_title="EcoPredictor")

with st.sidebar.title("Prediccion Inflacion"):
    with st.sidebar.title("Prediccion Inflacion"):
        st.markdown('Visualiza las prediciones de Inflacion basado en sus valores desde el 2018 hasta hoy 📈')            with st.sidebar.container():
             image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\eco Predictor.png')
             st.image(image, use_column_width=True)


header = st.container()
dataset = st.container()
features = st.container()
Inflacion = st.container()

with header:
    st.title("Prediccion Inflacion")

#Inflacion Grafico 

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
    layout = go.Layout(title='Inflacion Mensual % Historica', xaxis=dict(title='Fecha'), yaxis=dict(title='Valor'))

    # create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the chart
    st.write(fig, width=800, height=400)

#trainingIF.ipynb

def Inflacion():


    import pandas as pd
    import matplotlib.pyplot as plt
    import time 

    df = pd.read_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\inflacion.xlsx.pkl')


    # renaming for fbprophet
    df.rename(columns={'d':'ds'}, inplace=True)
    df.rename(columns={'v':'y'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()

    df.rename(columns={'Fecha':'ds'}, inplace=True)

    df = df[df['ds'] >= '2018-01-01']

    df = df.sort_values(by='ds')

    df['y'].round(2).describe()

    from prophet import Prophet

    prophet_model = Prophet()
    prophet_model.fit(df)

    future_dataset= prophet_model.make_future_dataframe(periods=1, freq='y') # Data para el proximo año
    future_dataset.tail()


    pred = prophet_model.predict(future_dataset)
    pred[['ds','yhat', 'yhat_lower', 'yhat_upper']].head() # only useful columns


    prophet_model.plot(pred)


    def fb_prophet_function(data, future_years, seasonality_name, seasonality_val,seasonality_fourier, **params):
        """
        Trains a fb prophet model on given hyperparameters and custom
        seasonality, predicts on future dataset, plot the results and
        return the model.
        """
        start= time.time()
        prophet_model = Prophet(**params)
        
        prophet_model.add_seasonality(name=seasonality_name, period=seasonality_val, fourier_order=seasonality_fourier)
            
        prophet_model.fit(data)
        
        future_dataset = prophet_model.make_future_dataframe(periods=future_years, freq='y')
        
        pred = prophet_model.predict(future_dataset)
        
        prophet_model.plot(pred, figsize=(15,7));
        plt.ylim(-500, 3000)
        plt.title(f"fourier order{seasonality_fourier}, seasonality time {seasonality_name}")
        plt.show()
        
        end = time.time()
        print(f"Total Execution Time {end-start} seconds")
        return prophet_model


    def plot_valid(validation_set, size, model):
        pred = model.predict(validation_set)
        temp = df[-size:].copy().reset_index()
        temp['pred']=pred['yhat']
        temp.set_index('ds')[['y', 'pred']].plot()
        plt.tight_layout();


    import time

    training_set = df
    validation_set = df 

    ten_years = fb_prophet_function(data=training_set, future_years=10, seasonality_name='10_years', seasonality_val=365*10, seasonality_fourier=100,seasonality_mode='additive')


    plot_valid(validation_set, 1000, ten_years)


    pred = pred[['ds', 'yhat']]


    validation_set = validation_set[['ds', 'y']]

    pred = pred[pred['ds'].isin(validation_set['ds'])]

    validation_set['ds'] = pd.to_datetime(validation_set['ds'])
    pred['ds'] = pd.to_datetime(pred['ds'])

    merged1 = pd.merge(pred, validation_set, on='ds', how='inner')

    validation_ds_y = merged1[['ds', 'y']]
    pred_ds_yhat = merged1[['ds', 'yhat']]


    validation_ds_y['ds'] = validation_ds_y['ds'].apply(lambda x: x.timestamp())
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].apply(lambda x: x.timestamp())


    validation_ds_y['ds'] = validation_ds_y['ds'].astype(float)
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].astype(float)


    import math
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    mae1 = mean_absolute_error(validation_ds_y, pred_ds_yhat)
    mse1 = mean_squared_error(validation_ds_y, pred_ds_yhat)
    rmse1 = math.sqrt(mean_squared_error(validation_ds_y, pred_ds_yhat))

    print("Mean Absolute Error: ", mae1)
    print("Mean Squared Error: ", mse1)
    print("Root Mean Squared Error: ", rmse1)


    training_set = df
    validation_set = df

    five_years_model = fb_prophet_function(data=training_set, future_years=10, seasonality_name='10_years', seasonality_val=365*10, seasonality_fourier=150,seasonality_mode='additive')


    plot_valid(validation_set, 1000, five_years_model)


    from prophet import Prophet

    five_years_model = Prophet(seasonality_mode='additive', seasonality_prior_scale=1, 
                            yearly_seasonality=True, weekly_seasonality=False, 
                            daily_seasonality=False)



    five_years_model.add_seasonality(name='1_years', period=365*1, fourier_order=100)


    import datetime

    today = datetime.datetime.now()
    next_year = today + datetime.timedelta(days=365)
    start_date = today.strftime("%Y-%m-%d")
    end_date = next_year.strftime("%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date, freq='M')
    next_year = pd.DataFrame({"ds": date_range})


    five_years_model.fit(training_set)

    prediction = five_years_model.predict(next_year)

    values = prediction['yhat']

    values_new = pd.DataFrame(values)

    values_new = values_new.rename(columns={'yhat':'Values'})

    # import the datetime library
    import datetime

    # define the start date (next month) and the number of months in the range
    start_date = (datetime.datetime.today() + datetime.timedelta(days=1)).replace(day=1)
    start_date = start_date.replace(month = (start_date.month % 12) + 1)
    num_months = 12

    # create the date range
    date_range = [start_date.date() + datetime.timedelta(days=30*i) for i in range(num_months)]

    # set the index of the dataframe to the date range
    values_new.index = date_range

    values_new.to_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newIF.xlsx")


    import pandas as pd
    import plotly.express as px
    import streamlit as st

    # Load the data from the xlsx file
    df = pd.read_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newIF.xlsx")

    # Plot the data using Plotly
    fig = px.line(df, x=df.index, y='Values')
    st.plotly_chart(fig)

if st.button("Predecir"):
    Inflacion()