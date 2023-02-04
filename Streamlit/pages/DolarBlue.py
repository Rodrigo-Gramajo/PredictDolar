import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo



st.set_page_config(page_title="Dolar Blue")

with st.sidebar.title("Dolar Blue"):
    with st.sidebar.title("Dolar Blue"):
            st.markdown('Visualiza los datos historicos de las 6 variables que tienen mas del 80% de correlacion :sunglasses:')


header = st.container()
dataset = st.container()
features = st.container()
Dolar_Blue = st.container()

with header:
    st.title("Dolar Blue")

#DolarApiBlue.ipynb

def Dolar_Blue():

    import requests
    import pandas as pd
    import datetime

    # Fecha actual
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")

    # Fecha de inicio 
    start_date = '01-01-2010'

    # Construimos la url con las fechas
    url = f"https://mercados.ambito.com//dolar/informal/historico-general/{start_date}/{current_date}"

    response = requests.get(url)
    data = response.json()

    # In[2]:


    # Convertimos JSON a unDataFrame
    df = pd.DataFrame(data)

   

    # In[3]:


    #Cambiamos nombres de las columnas

    df.rename(columns={0:'Fecha', 1:'Compra', 2:'Venta'}, inplace=True)
    df.set_index('Fecha', inplace=True)
    


    # In[4]:


    #Cambiamos nombres de las columnas y ordenamos

    df.rename(columns={0:'Fecha', 1:'Compra', 2:'Venta'}, inplace=True)
    


    # In[5]:


    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'Venta'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()


    # In[6]:


    df.drop(0, inplace=True)
   

    # In[7]:


    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%Y')
    df['Compra'] = df['Compra'].str.replace(',', '.').astype(float)
    df['Venta'] = df['Venta'].str.replace(',', '.').astype(float)
    df.set_index('Fecha', inplace=True)
  


    # In[8]:


    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'Venta'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()


    # In[9]:


    df.to_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PrecioBlueDIario.xlsx")

    #BlueDiarioApi.ipynb

    import json
    import requests

    url = "https://www.dolarsi.com/api/api.php?type=dolar"
    response = requests.get(url)
    data = json.loads(response.text)

    print(data)


    # In[2]:


    import pandas as pd

    df = pd.DataFrame(data)
    df = df['casa'].apply(pd.Series)

    print(df)


    # In[3]:


    df = df[["nombre","venta"]]
    print(df)


    # In[4]:


    pd.DataFrame(df)


    # In[5]:


    import datetime
    import pandas as pd

    today = datetime.datetime.today().strftime('%Y-%m-%d')
    value = df[df['nombre'] == 'Blue']['venta'].values[0]
    value = format(float(value.replace(',', '')), '.1f')
    df = pd.DataFrame({'Fecha': [today], 'Compra': [0.0], 'Venta': [float(value.replace('.', ''))]})


    # In[6]:


    df['Venta'] = df['Venta'].astype(int) / 10000
    df['Venta'] = df['Venta'].apply(lambda x: "{:.1f}".format(x))


    # In[7]:


    df.to_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PrecioBlueDIario.xlsx', index=False)


    # In[8]:


    import pandas as pd

    # Load the existing Excel file into a dataframe
    df2 = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosBlue.xlsx')

    # Append the first row of the dataframe to the existing dataframe
    df2 = df2.append(df.iloc[0, :], ignore_index=True)

    # Save the updated dataframe to the Excel file
    df2.to_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosBlue.xlsx', index=False)


    #visualizationsBlue.ipynb


    import pandas as pd

    df = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\PreciosBlue.xlsx')
    print(df)


    # In[2]:


    df = pd.DataFrame(df)
 
    # In[5]:


    df.drop(df.columns[0], axis=1, inplace=True)
    df.rename(columns={0: 'Fecha', 1: 'Compra', 2: 'Venta'}, inplace=True)
  


    df['Compra'] = pd.to_numeric(df['Compra'])
    df['Venta'] = pd.to_numeric(df['Venta'])


    # In[8]:


    df.to_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\PreciosBlue.xlsx.pkl')


    # In[9]:


    df.set_index('Fecha', inplace=True)
    df.index = pd.to_datetime(df.index, format='%d-%m-%Y')


    # In[10]:


    df.resample(rule='M').mean().plot(); # Historico


    # In[11]:


    df.resample(rule='AS').mean().plot(); # AS Primer año


    # In[12]:


    ax = df.resample(rule='AS').mean().plot.bar(figsize=(12,6))
    ax.set(title='Promedio valor Dolar')


    # In[13]:


    df['DOLLAR_12M'] = df['Venta'].rolling(window=365).mean()
    df['DOLLAR_5Y'] = df['Venta'].rolling(window=365*5+1).mean()
    df[['Venta','DOLLAR_12M','DOLLAR_5Y']].plot(figsize=(18,6))


    # In[14]:


    ax = df.resample(rule='AS').mean().plot.bar(figsize=(12,6))


    # In[15]:


    ax = df['Venta'].resample(rule='M').mean().plot(figsize=(15,8), label='Resample MS') # monthly resampled mean
    ax.autoscale(tight=True)
    df.rolling(window=30).mean()['Venta'].plot(label='Rolling window=30') # monthly rolling windows/moving average
    ax.set(title='Precio promedio del Dolar')
    ax.legend()


    # In[16]:


    from datetime import datetime

    ax = df['Venta'].resample(rule='M').mean().plot(xlim=['2018-01-01', datetime.now()], figsize=(15,8), label='Resample MS')
    ax.autoscale(tight=True)
    df.rolling(window=30).mean()['Venta'].plot(xlim=['2018-01-01',datetime.now()],label='Rolling window=30')
    ax.set(title='Precio promedio del Dolar')
    ax.legend()


    # In[17]:


    df.resample(rule='1y').mean().plot() #1y = 1 years


    # In[18]:


    df.resample(rule='5y').mean().plot() #5y = 5 years


    # In[19]:


    df.resample(rule='10y').mean().plot()


    # In[20]:


    df.resample(rule='15y').mean().plot()


    #trainingBlue.ipynb

    import pandas as pd
    import matplotlib.pyplot as plt
    import time 

    df = pd.read_pickle(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\pkl\PreciosBlue.xlsx.pkl')


    # In[2]:


    # renaming for fbprophet
    df.rename(columns={'Fecha':'ds'}, inplace=True)
    df.rename(columns={'Venta':'y'}, inplace=True)
    df.reset_index(inplace=True)
    df.head()


    # In[3]:


    df.rename(columns={'Fecha':'ds'}, inplace=True)


    # In[4]:


    df = df[df['ds'] >= '2018-01-01']



    # In[5]:


    df.drop(['Compra'], axis=1)


    # In[6]:


    df = df.sort_values(by='ds')


    # In[9]:


    from prophet import Prophet

    prophet_model = Prophet()
    prophet_model.fit(df)


    # In[10]:


    future_dataset= prophet_model.make_future_dataframe(periods=1, freq='y') # Data para el proximo año
    future_dataset.tail()


    # In[11]:


    pred = prophet_model.predict(future_dataset)



    # In[12]:


    prophet_model.plot(pred)


    # In[13]:


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


    # In[14]:


    def plot_valid(validation_set, size, model):
        pred = model.predict(validation_set)
        temp = df[-size:].copy().reset_index()
        temp['pred']=pred['yhat']
        temp.set_index('ds')[['y', 'pred']].plot()
        plt.tight_layout();


    # In[15]:


    import time

    training_set = df
    validation_set = df 

    ten_years = fb_prophet_function(data=training_set, future_years=10, seasonality_name='10_years', seasonality_val=365*10, seasonality_fourier=600,seasonality_mode='additive')


    # In[16]:


    plot_valid(validation_set, 1000, ten_years)


    # In[17]:


    pred = pred[['ds', 'yhat']]


    # In[18]:


    validation_set = validation_set[['ds', 'y']]



    # In[19]:


    pred = pred[pred['ds'].isin(validation_set['ds'])]


    # In[20]:


    merged1 = pd.merge(pred, validation_set, on='ds', how='inner')

    validation_ds_y = merged1[['ds', 'y']]
    pred_ds_yhat = merged1[['ds', 'yhat']]


    # In[21]:


    validation_ds_y['ds'] = validation_ds_y['ds'].apply(lambda x: x.timestamp())
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].apply(lambda x: x.timestamp())


    # In[22]:


    validation_ds_y['ds'] = validation_ds_y['ds'].astype(float)
    pred_ds_yhat['ds'] = pred_ds_yhat['ds'].astype(float)


    # In[23]:


    import math
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    mae1 = mean_absolute_error(validation_ds_y, pred_ds_yhat)
    mse1 = mean_squared_error(validation_ds_y, pred_ds_yhat)
    rmse1 = math.sqrt(mean_squared_error(validation_ds_y, pred_ds_yhat))

    print("Mean Absolute Error: ", mae1)
    print("Mean Squared Error: ", mse1)
    print("Root Mean Squared Error: ", rmse1)


    # In[24]:


    df = df[df['ds'] >= '2021-01-01']


    # In[25]:


    training_set = df
    validation_set = df

    five_years_model = fb_prophet_function(data=training_set, future_years=5, seasonality_name='5_years', seasonality_val=365*5, seasonality_fourier=500,seasonality_mode='additive')


    # In[26]:


    plot_valid(validation_set, 1000, five_years_model)


    # In[27]:


    from prophet.diagnostics import cross_validation, performance_metrics

    model = Prophet()
    model.fit(df)

    df_cv = cross_validation(model, initial='360 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv, rolling_window=0.1) 
    

    # In[28]:


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


    # In[29]:


    from prophet import Prophet

    five_years_model = Prophet(seasonality_mode='additive', seasonality_prior_scale=1, 
                            yearly_seasonality=True, weekly_seasonality=False, 
                            daily_seasonality=False)


    # In[30]:


    five_years_model.add_seasonality(name='1_years', period=365*1, fourier_order=90)


    # In[31]:


    import datetime

    today = datetime.datetime.now()
    next_month = today + datetime.timedelta(days=90)
    start_date = today.strftime("%Y-%m-%d")
    end_date = next_month.strftime("%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date)
    next_month = pd.DataFrame({"ds": date_range})


    # In[32]:


    five_years_model.fit(training_set)

    prediction = five_years_model.predict(next_month)


    # In[33]:


    values = prediction['yhat']

    values_new = pd.DataFrame(values)

    values_new = values_new.rename(columns={'yhat':'Values'})


    # In[34]:


    # import the datetime library
    import datetime

    # define the start date (today + 1) and the number of days in the range
    start_date = datetime.datetime.today() + datetime.timedelta(days=1)
    num_days = len(values_new.index)

    # create the date range
    date_range = [start_date.date() + datetime.timedelta(days=i) for i in range(num_days)]

    # set the index of the dataframe to the date range
    values_new.index = date_range

    # In[35]:


    values_new.to_excel(r"C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newBlue.xlsx")

    df = pd.read_excel(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\xlsx\values_newBlue.xlsx')
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

    layout = go.Layout(title='Prediccion Dolar Blue', xaxis=dict(title='Fecha'), yaxis=dict(title='Valores'))
    fig = go.Figure(data=data, layout=layout)

    st.write(fig)

if st.button("Predict"):
    Dolar_Blue()

