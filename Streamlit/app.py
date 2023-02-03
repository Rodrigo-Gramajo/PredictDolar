import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("PBI per Capita Estimador")

with features:
    st.header('Variables')
    año = st.number_input("Año", min_value=2010, max_value=2021, value=2010)
    cerveza = st.number_input("Consumo Cerveza Per Capita")
    fertilizante = st.number_input("Consumo Fertilizante")
    pescado = st.number_input("Consumo Pescados Per Capita")
    recursos_fosiles = st.number_input("Consumo Recursos Fosiles Per Capita")
    fruta = st.number_input("Consumo Fruta Per Capita")

with model_training:

    df = pd.read_csv(r'C:\Users\rodri\OneDrive\Escritorio\Digital\Dolar V4\PredictDolar\data\csv\PBI.csv')
    features = ['Año', 'Consumo Cerveza Per Capita', 'Consumo Fertilizante', 'Consumo Pescados Per Capita', 'Consumo Recursos Fosiles Per Capita', 'Consumo Fruta Per Capita']
    X = df[features]
    y = df['PBI per Capita']

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    
    if st.button("Predecir"):
        input_values = [año, cerveza, fertilizante, pescado, recursos_fosiles, fruta]
        prediction = model.predict([input_values])
        st.write("PBI per Capita: ", prediction[0])
