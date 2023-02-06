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

with st.sidebar.title("Prediccion Dolar Oficial"):
    with st.sidebar.title("Prediccion Dolar Oficial"):
        st.markdown('Visualiza las prediciones del Dolar Oficial basado en sus valores desde el 2018 hasta hoy 💰')            
        with st.sidebar.container():
              image = Image.open(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\Images\eco Predictor.png')
              st.image(image, use_column_width=True)



header = st.container()
dataset = st.container()
features = st.container()
Dolar_OF = st.container()

with header:
    st.title("Prediccion Dolar Oficial")

#Dolares 

import plotly.express as px
import pandas as pd

# Read the data from the xlsx file
df = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx', engine='openpyxl')

# Create a plot with two lines, one for the 'Compra' column in blue and one for the 'Venta' column in red
fig = px.line(df, x='Fecha', y='Compra')
fig.add_scatter(x=df['Fecha'], y=df['Venta'], mode='lines', line=dict(color='red', width=2), showlegend=False)

# Set the title of the plot
fig.update_layout(title={
        'text': "Dolar Oficial Compra & Venta Historico",
        'font': {'size': 24},
        'x': 0.5,
        'xanchor': 'center',
        'y': 0.95
    },
    xaxis_title="Fecha",
    yaxis_title="Precio"
)

st.write(fig)

#DolarApiOF.ipynb

def Dolar_OF():

    import requests
    import pandas as pd
    import datetime

    # Fecha actual
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")

    # Fecha de inicio 
    start_date = '01-01-2010'

    # Construimos la url con las fechas
    url = f"https://mercados.ambito.com//dolar/dolar-oficial/historico-general/{start_date}/{current_date}"

    response = requests.get(url)
    data = response.json()

    # Convertimos JSON a unDataFrame
    df = pd.DataFrame(data)

    #Cambiamos nombres de las columnas

    df.rename(columns={0:'Fecha', 1:'Compra', 2:'Venta'}, inplace=True)
    df.set_index('Fecha', inplace=True)
    

    #Cambiamos nombres de las columnas y ordenamos

    df.rename(columns={0:'Fecha', 1:'Compra', 2:'Venta'}, inplace=True)

    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'Venta'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()


    df.drop(0, inplace=True)
   

    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%Y')
    df['Compra'] = df['Compra'].str.replace(',', '.').astype(float)
    df['Venta'] = df['Venta'].str.replace(',', '.').astype(float)
    df.set_index('Fecha', inplace=True)
  

    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'Venta'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()

    df.to_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx")
    
#OFDiarioApi.ipynb

    import json
    import requests

    url = "https://www.dolarsi.com/api/api.php?type=dolar"
    response = requests.get(url)
    data = json.loads(response.text)

    print(data)


    import pandas as pd

    df = pd.DataFrame(data)
    df = df['casa'].apply(pd.Series)

    df = df[["nombre","venta"]]
    print(df)

    pd.DataFrame(df)

    import datetime
    import pandas as pd

    import datetime
    import pandas as pd

    today = datetime.datetime.today().strftime('%Y-%m-%d')
    value = df[df['nombre'] == 'Oficial']['venta'].values[0]
    value = format(float(value.replace(',', '')), '.1f')
    df = pd.DataFrame({'Fecha': [today], 'Compra': [0.0], 'Venta': [float(value.replace('.', ''))]})
    df['Venta'] = df['Venta'].astype(int) / 10000
    df['Venta'] = df['Venta'].apply(lambda x: "{:.1f}".format(x))
    

    df.to_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PrecioBlueDIario.xlsx', index=False)
    import pandas as pd

    # Load the existing Excel file into a dataframe
    df2 = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx')

    # Append the first row of the dataframe to the existing dataframe
    df2 = df2.append(df.iloc[0, :], ignore_index=True)

    # Save the updated dataframe to the Excel file
    df2.to_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx', index=False)



#visualizationsOF.ipynb


    import pandas as pd

    df = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx')
    print(df)

    df = pd.DataFrame(df)
 
 
    df.drop(df.columns[0], axis=1, inplace=True)
    df.rename(columns={0: 'Fecha', 1: 'Compra', 2: 'Venta'}, inplace=True)
  


    df['Compra'] = pd.to_numeric(df['Compra'])
    df['Venta'] = pd.to_numeric(df['Venta'])


    df.to_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\PreciosOF.xlsx.pkl')

#trainingOF.ipynb

    import pandas as pd
    import matplotlib.pyplot as plt
    import time 

    df = pd.read_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\PreciosOF.xlsx.pkl')

    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'y'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()


    df.rename(columns={'Fecha':'ds'}, inplace=True)

    df = df[df['ds'] >= '2018-01-01']

    df.drop(['Compra'], axis=1)

    df = df.sort_values(by='ds')

    from prophet import Prophet

    prophet_model = Prophet()
    prophet_model.fit(df)


    future_dataset= prophet_model.make_future_dataframe(periods=1, freq='y') # Data para el proximo año
    future_dataset.tail()


    pred = prophet_model.predict(future_dataset)


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

    ten_years = fb_prophet_function(data=training_set, future_years=10, seasonality_name='10_years', seasonality_val=365*10, seasonality_fourier=600,seasonality_mode='additive')



    plot_valid(validation_set, 1000, ten_years)


    pred = pred[['ds', 'yhat']]


    validation_set = validation_set[['ds', 'y']]


    pred = pred[pred['ds'].isin(validation_set['ds'])]


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



    df = df[df['ds'] >= '2021-01-01']



    training_set = df
    validation_set = df

    five_years_model = fb_prophet_function(data=training_set, future_years=5, seasonality_name='5_years', seasonality_val=365*5, seasonality_fourier=500,seasonality_mode='additive')


    plot_valid(validation_set, 1000, five_years_model)



    from prophet.diagnostics import cross_validation, performance_metrics

    model = Prophet()
    model.fit(df)

    df_cv = cross_validation(model, initial='360 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv, rolling_window=0.1) 
    


    from sklearn.metrics import mean_squared_error
    pred = pred[['ds', 'yhat']]


    validation_set = validation_set[['ds', 'y']]

   
    pred = pred[pred['ds'].isin(validation_set['ds'])]

  
    merged2 = pd.merge(pred, validation_set, on='ds', how='inner')

    validation_ds_y = merged2[['ds', 'y']]
    pred_ds_yhat = merged2[['ds', 'yhat']]

    validation_ds_y['ds'] = validation_ds_y['ds'].apply(lambda x: x.timestamp())
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].apply(lambda x: x.timestamp())
    validation_ds_y['ds'] = validation_ds_y['ds'].astype(float)
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].astype(float)
    import math

    mae2 = mean_absolute_error(validation_ds_y, pred_ds_yhat)
    mse2 = mean_squared_error(validation_ds_y, pred_ds_yhat)
    rmse2 = math.sqrt(mean_squared_error(validation_ds_y, pred_ds_yhat))

    print("Mean Absolute Error: ", mae2)
    print("Mean Squared Error: ", mse2)
    print("Root Mean Squared Error: ", rmse2)


    from prophet import Prophet

    five_years_model = Prophet(seasonality_mode='additive', seasonality_prior_scale=1, 
                            yearly_seasonality=True, weekly_seasonality=False, 
                            daily_seasonality=False)


    five_years_model.add_seasonality(name='1_years', period=365*1, fourier_order=90)

    import datetime

    today = datetime.datetime.now()
    next_month = today + datetime.timedelta(days=90)
    start_date = today.strftime("%Y-%m-%d")
    end_date = next_month.strftime("%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date)
    next_month = pd.DataFrame({"ds": date_range})

    five_years_model.fit(training_set)

    prediction = five_years_model.predict(next_month)


    values = prediction['yhat']

    values_new = pd.DataFrame(values)

    values_new = values_new.rename(columns={'yhat':'Values'})


    # import the datetime library
    import datetime

    # define the start date (today + 1) and the number of days in the range
    start_date = datetime.datetime.today() + datetime.timedelta(days=1)
    num_days = len(values_new.index)

    # create the date range
    date_range = [start_date.date() + datetime.timedelta(days=i) for i in range(num_days)]

    # set the index of the dataframe to the date range
    values_new.index = date_range

    values_new.to_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newOf.xlsx")

    import streamlit as st
    import pandas as pd

    # Read the data from the xlsx files
    precios_blue = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosOF.xlsx')
    values_new_blue = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newOF.xlsx')

    # Get the penultimate day of PreciosBlue.xlsx
    penultimate_day_precios = precios_blue['Venta'].iloc[1].astype(int)

    # Get the last day of PreciosBlue.xlsx
    last_day_precios = precios_blue['Venta'].iloc[-1].astype(int)

    # Get the first day of values_newBlue.xlsx
    first_day_values_new = values_new_blue['Values'].iloc[0].astype(int)

    # Split the screen into three columns
    col1, col2, col3 = st.columns(3)

    # Display the penultimate day of PreciosBlue.xlsx in col1
    col1.metric("Ayer", penultimate_day_precios)

    # Display the last day of PreciosBlue.xlsx in col2
    col2.metric("Ahora", last_day_precios)

    # Display the first day of values_newBlue.xlsx in col3
    col3.metric("Mañana", first_day_values_new)
    df = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newOf.xlsx')
    df = df.rename(columns={'Unnamed: 0': 'Fecha'})
    x = df['Fecha']
    y = df['Values']

    data = []
    for i in range(1, len(y)):
        if y[i] > y[i-1]:
            color = 'green'
        else:
            color = 'red'
        trace = go.Scatter(x=x[i-1:i+1], y=y[i-1:i+1], mode='lines', line=dict(color=color, width=1), showlegend=False)
        data.append(trace)

    layout = go.Layout(xaxis=dict(title='Fecha'), yaxis=dict(title='Valores'))
    fig = go.Figure(data=data, layout=layout)

    st.write(fig)

if st.button("Predecir"):
    Dolar_OF()

