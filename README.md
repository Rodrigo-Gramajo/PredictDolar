### Proyecto Final Integrador

### Autor: Rodrigo Gramajo

Tema de investigación: Variables Macro & Micro económicas de Argentina

## Diseñar una aplicación de Machine Learning para predecir

## variables Macro & Micro economicas de Argentina

### Introducción:

¿Cuál es el objetivo de este proyecto?

Se propone desarrollar modelos de aprendizaje automático para analizar, identificar
y predecir valores futuros de algunas de las variables con más relevancia a nivel
económico de Argentina.

En este caso de estudio, me centro en el análisis de diferentes fuentes de
información citadas más adelante de esta presentación.

### Contexto del Proyecto:

En la actualidad, la economía juega un papel fundamental en el desarrollo de una
sociedad. Para asegurar su crecimiento y estabilidad, es importante tener una
buena comprensión de las tendencias y patrones económicos. La inteligencia
artificial y el aprendizaje automático, combinados con la ciencia de datos, brindan
una nueva forma de realizar predicciones económicas con una mayor precisión y
eficiencia.

El uso de técnicas de aprendizaje automático permite a los economistas analizar
grandes cantidades de datos y descubrir patrones y tendencias que de otra manera
podrían pasar desapercibidos. Esto permite una toma de decisiones más informada
y una mejor planificación a largo plazo en el mundo empresarial y gubernamental.

Además, el uso de modelos predictivos basados en datos también permite a los
economistas responder de manera más eficiente a situaciones imprevistas, como
crisis económicas o cambios en el mercado. Con la capacidad de realizar

predicciones precisas y en tiempo real, las empresas y los gobiernos pueden tomar
medidas proactivas para mitigar el impacto de eventos económicos adversos.

En resumen, la combinación de inteligencia artificial, aprendizaje automático y
ciencia de datos está transformando la forma en que se hacen las predicciones
económicas. Permiten una comprensión más profunda y precisa de las tendencias y
patrones económicos, lo que se traduce en una toma de decisiones más informada
y un futuro económico más estable y predecible.

### Ciclo de trabajo:

```
● Recolección de datos (obtener información de fuentes primarias)
● Extracción de información (Procesamiento de datos primarios para tenerlos
en el formato requerido)
● Predicción de la información
● Visualización de las predicciones y fuentes primarias
```
### Propuesta de valor:

Mejor comprensión de las tendencias económicas: Al utilizar técnicas avanzadas de
aprendizaje automático, se pueden analizar grandes cantidades de datos para
identificar patrones y tendencias que de otra manera podrían pasar desapercibidos.
Esto brinda una visión más completa y precisa del panorama económico.

Toma de decisiones más informada: Al tener acceso a predicciones precisas y
actualizadas, las empresas y los gobiernos pueden tomar decisiones más
informadas y planificar su futuro de manera más eficiente.

Mayor estabilidad económica: Al poder predecir y responder a situaciones
imprevistas, como crisis económicas o cambios en el mercado, las empresas y los
gobiernos pueden tomar medidas proactivas para mitigar el impacto de eventos
económicos adversos.

Ahorro de tiempo y recursos: Al utilizar modelos de aprendizaje automático, se
puede analizar una gran cantidad de datos en un corto período de tiempo, lo que
resulta en un ahorro de tiempo y recursos.


Mejora continua: Al tener acceso a datos actualizados, los modelos de aprendizaje
automático pueden ser actualizados y mejorados constantemente, garantizando una
mejora continua en la precisión de las predicciones.

En resumen, el desarrollo de modelos de aprendizaje automático para analizar,
identificar y predecir valores futuros de variables económicas en Argentina brinda
una mayor comprensión de las tendencias económicas, toma de decisiones más
informada, mayor estabilidad económica, ahorro de tiempo y recursos, y mejora
continua.

### Modelo predictivo - Objetivo

**¿Qué data recolectamos?**

Fuentes de información primaria*

**API:**
https://ourworldindata.org/
https://mercados.ambito.com//dolar
https://www.dolarsi.com/api/api.php
https://api.estadisticasbcra.com/

Recolectamos Información estadística **Histórica y Actualizada** (en el momento) de
diferentes fuentes.

**¿Qué vamos a predecir?**

```
● Predicción de los próximos 7 días * del valor del DólarOficial & Blue
● Predicción Brecha cambiaria para los próximos 7 días *
● Predicción de los próximos 12 meses de la inflación
● Predicción del PBI per Cápita (basándonos en las 6 variables más
correlativas del modelo*)
```
_Si bien el modelo de predicción nos devuelve los 90 días, por cuestiones de
varianza disponibilizamos los próximos 7 días*_


**¿Qué vamos a disponibilizar?**

```
● Gráfico de PBI per Cápita, Producción Agrícola $ MM, Suscripciones a
Celulares, Consumo de Cerveza per Cápita, Consumo de Cerveza per
Cápita, Expectativa de Vida, Consumo de recursos Fósiles per Cápita > Año
2000
● Gráfico de Hitos Económicos > 2010
● Gráfico de Inflación Mensual > 2010
● Gráfico de Dólar Oficial & Dolar Blue > 2010
```
**¿Cómo se recolectan los datos que alimentan al modelo?**

Se trabajaron con 2 modelos predominantes:

```
● Para la Predicción del PBI per Cápita, opte por el modelo que obtuvo mejores
valores en las pruebas luego de haber probado Gradient Boosting, SVM,
linear regression, Random Forest, el mismo es un algoritmode aprendizaje
automático de ensamble que se utiliza en tareas de clasificación y regresión.
Se basa en la creación de múltiples árboles de decisión y toma una decisión
basada en la mayoría de los votos de los árboles individuales. Este algoritmo
es conocido por ser resistente a la sobre-adaptación y tiene una alta precisión
en problemas complejos.
```

```
● Para la Predicción del valor del Dólar Oficial & Blue, inflación, opte por el
modelo Prophet el cual es un algoritmo de aprendizajeautomático de series
de tiempo desarrollado por Facebook. Se utiliza para hacer predicciones
sobre series de tiempo, incluyendo tendencias, fluctuaciones estacionales y
eventos puntuales. Prophet utiliza un enfoque basado en modelos para hacer
sus predicciones, lo que resulta en una alta precisión en una amplia variedad
de aplicaciones de series de tiempo. Además, es fácil de usar y puede
manejar datos con tasas de cambio no lineales y faltantes de manera
efectiva, y decidí utilizarlo ya que la información primaria obtenida era series
de tiempo, y tenían una fluctuación estacional importante, este modelo es
ideal para manejar este tipo de información (modelo muy utilizado para las
predicciones de acciones).
```
### Flujo de Trabajo

**https://drive.google.com/file/d/1S9XL5PJAZfnFu4GtDI8dQzBIFPc9lJgZ/view?usp=sharing**

#### IPYNB:

En una primera instancia se empezó a trabajar con Jupyter Notebook para poder
tener un mejor entendimiento del proyecto:

1. **BlueDiarioApi.ipynb** : Genera un request a la API
    "https://www.dolarsi.com/api/api.php?type=dólar", la cual devuelve las
    cotizaciones del momento del Dólar, luego crea un DF del JSON, y filtra para
    conocer las cotizaciones del Blue, Actualiza los XLSX de los códigos:
    DolarApiBlue.ipynb

**2. Brecha.ipynb** : Lee los valores de los XLSX: values_newOf.xlsx,values
    newBlue.xlsx, luego crea junta en ambos DF en uno solo el cual tiene los
    valores del Dólar Blue y Dólar Oficial Precedidos para los próximos 90 días,
    Genera unas limpiezas del DF, calcula la brecha entre ambos valores todos
    los días y genera un gráfico para visualizarlo
**3. DolarApiBlue.ipynb** : Genera un request a la API
    "https://mercados.ambito.com//dolar/informal/historico-general/{start_date}/{cu
    rrent_date}", la cual devuelve las cotizaciones Históricas del Dólar Blue,
    Luego crea un DF del JSON, limpia y ordena los datos para generar un buen
    DF en formato de Base de Datos, exporta el DF en un archivo XLSX:
    Precio Blue.xlsx.
**4. DolarApiOF.ipynb** : Genera un request a la API
    "https://mercados.ambito.com//dolar/dolar-oficial/historico-general/{start_date}
    /{current_date}", la cual devuelve las cotizaciones Históricas del Dólar Oficial,
    Luego crea un DF del JSON, limpia y ordena los datos para generar un buen
    DF en formato de Base de Datos, Exporta el DF en un archivo XLSX:
    PreciosO.xlsx.
**5. EconomiaApi.ipynb** : Generamos 2 request a la API delBCRA
    https://api.estadisticasbcra.com, milestones, inflaciónmensual oficial, Luego
    creamos un DF del JSON para los Hitos y otro para a Inflación (Históricos),
    además importamos los XLSX: PreciosBlue, PreciosOF, Generamos los
    gráficos Históricos para estas métricas: Inflación Mensual Histórica, Dólar
    Blue Histórico, Dólar Oficial Histórico, Hitos Histórico (Esta API funciona con
    TOKEN de acceso*).
**6. Merged.ipynb** : Generamos nuestro propio DS desde 33archivos CSV
    diferentes, todos fueron obtenidos desde esta pagina:
    https://ourworldindata.org/, la cual nos brindó variasmétricas a nivel Global,
    Con las 33 variables de todos los DF generamos a base de filtros nuestro DF
    que es para Argentina y lo ordenamos por el Año, Luego cambiamos el
    nombre de las columnas para que sea más legible y empezamos a hacer un
    análisis para dejar listo el DF, Empezamos a trabajar con los datos > 2000, y
    eliminamos las columnas con más del 20% de NaN, luego verificamos que la
    columna PBI per Cápita contenga a información correcta, Validamos entre las
    27 variables que dejamos la correlación entre ellas con un mapa de calor y
    luego generamos 9 gráficos para las variables con mayor correlación con PBI
    per Cápita, Además generamos todos los arreglos previos a poder procesar
    el DF con sklearn, utilizamos un 20% para las pruebas y un 80% para la
    práctica.

```
```
```
# linear regression model
Mean Absolute Error (train): 0.
Mean Absolute Error (test): 1.20256795681 14722
```
```
# SVM model
mae_train: 0.
mae_test: 0.
```
```
# Gradient Boosting model
mae_train: 1.
mae_test: 0.
```
```
# Random Forest model
mae_train: 0.
mae_test: 0.21032915921 19624
```
```
Utilice los Hiperparametros :
```
```
fit_intercept : True/False, para quedar o quitar laconstante de nuestro
modelo.
```
```
normalize : True/False, para normalizar los datos ono.
```
Generamos un decision tree model y con data de ejemplo validamos si el
modelo puede predecir el PIB per Cápita.


_Analisis de Correlacion entre las 33 variables del modelo*_

_Variables con más del 80% de correlación con el PBI per Cápita*_


**7. OFDiarioApi.ipynb** : Genera un request a la API
    "https://www.dolarsi.com/api/api.php?type=dólar", la cual devuelve las
    cotizaciones del momento del Dólar, luego crea un DF del JSON, y filtra para
    conocer las cotizaciones del Oficial, Actualiza los XLSX de los códigos:
    Dolor piOF.
**8. Training Blue.ipynb** : Leemos el archivo PKL generadopor visualisations
    Blue.ipynb, empezamos el proceso para poder procesarlo con fb prophet ,
    Importamos Prophet para poder empezar con nuestro modelo de análisis de
    tiempo y generamos dos pruebas (10 años y 5 años) ambas con dos
    seasonality fourier (tendencia al cambio) diferentes:

```
Vamos a trabajar con el modelo de 5 Años, y le bajamos el fourier order para
tener una mejor estimación, le pedimos que nos genere un DS para los
próximos 90 días del valor del Dólar y lo graficamos
```
```
Entrenamiento con fbprophet*
```

**9. trainingIF.ipynb** : Leemos el archivo PKL generado porvisualisations.ipynb,
    empezamos el proceso para poder procesarlo con fb prophet ,Importamos
    Prophet para poder empezar con nuestro modelo de análisis de tiempo y
    generamos dos pruebas (10 años y 5 años) ambas con dos seasonality
    fourier (tendencia al cambio) diferentes

Ambos modelos son iguales porque contamos con data desde el 2018, igualmente
decido trabajar con el modelo de 5 Años ya que tiene otro fourier order para tener
una mejor estimación, le pedimos que nos genere un DS para los próximos 12
meses de la inflación.

_Entrenamiento con fbprophet*_

```
10.trainingOF.ipynb : Leemos el archivo PKL generadopor visualisations.ipynb,
empezamos el proceso para poder procesarlo con fb prophet, importamos
Prophet para poder empezar con nuestro modelo de análisis de tiempo y
generamos dos pruebas (10 años y 5 años) ambas
```

```
con dos seasonality fourier (tendencia al cambio) diferentes:
```
```
Vamos a trabajar con el modelo de 5 Años, y le bajamos el fourier order para
tener una mejor estimación, le pedimos que nos genere un DS para los
próximos 90 días del valor del dólar y lo graficamos.
```
```
Entrenamiento con fbprophet*
```
**11.vizualtisationBlue.ipynb** : Leemos el archivo XLSX generadopor Blue Diario
Api.ipynb, visualizamos los datos de DS para validar que podamos procesarlo
y ver los datos Históricos, vemos los promedios de los valores en un gráfico,
para descartar que haya valores extremos, también analizamos los rolling
windows de 5 Años y 12 meses, Generamos el archivo pickle:
Precio Blue.xlsx.pkl

12. **vizualtisationOF.ipynb** : Leemos el archivo XLSX generadopor
    BlueOFApi.ipynb, visualizamos los datos de DS para validar que podamos


```
procesarlo y ver los datos Históricos, vemos los promedios de los valores en
un gráfico, para descartar que haya valores extremos, también analizamos
los rolling windows de 5 Años y 12 meses, Generamos el archivo pickle:
PreciosOF.xlsx.pkl.
```
**13.vizualtisationIF.ipynb** : Leemos el archivo XLSX generadopor
Inflacion.ipynb, visualizamos los datos de DS para validar que podamos
procesarlo y ver los datos Históricos, vemos los promedios de los valores en
un gráfico, para descartar que haya valores extremos, también analizamos
los rolling windows de 5 Años y 12 meses, Generamos el archivo pickle:
Inflacion.xlsx.pkl


### Pipelines:

```
Inflación
```

_Dólares_


_PBI per Cápita_


### Fuentes de información primaria*

https://ourworldindata.org/grapher/gas-consumption-by-country?tab=chart&country=ARG
https://ourworldindata.org/grapher/beer-consumption-per-person?tab=chart&country=ARG
https://ourworldindata.org/grapher/fertilizer-consumption-usda?country=~ARG
https://ourworldindata.org/grapher/fish-and-seafood-consumption-per-capita?tab=chart&country=ARG
https://ourworldindata.org/grapher/fossil-fuel-primary-energy?tab=chart&country=ARG
https://ourworldindata.org/grapher/fruit-consumption-per-capita?tab=chart&country=ARG
https://ourworldindata.org/grapher/per-capita-egg-consumption-kilograms-per-year?tab=chart&country
=ARG
https://ourworldindata.org/graper/vegetable-consumption-per-capita?tab=chart&country=ARG
https://ourworldindata.org/grapher/wine-consumption-per-person?tab=chart&country=ARG
https://ourworldindata.org/grapher/share-of-education-in-government-expenditure?tab=chart&country
=ARG
https://ourworldindata.org/grapher/health-expenditure-government-expenditure?tab=chart&country=A
RG
https://ourworldindata.org/grapher/industry-share-of-total-emplyoment?tab=chart&country=ARG
https://ourworldindata.org/grapher/share-of-employee-compensation-in-public-spending?tab=chart&co
untry=ARG
https://ourworldindata.org/grapher/share-food-exports?tab=chart&country=ARG
https://ourworldindata.org/grapher/share-of-services-in-total-exports?tab=chart&country=ARG
https://ourworldindata.org/grapher/exports-of-goods-and-services-constant-2010-us?tab=chart&countr
y=ARG
https://ourworldindata.org/grapher/share-food-exports?tab=chart&country=ARG
https://ourworldindata.org/grapher/share-manufacture-exports?tab=chart&country=ARG
https://ourworldindata.org/grapher/gdp-per-capita-worldbank?tab=chart&country=ARG
https://ourworldindata.org/grapher/population-density?tab=chart&country=ARG
https://ourworldindata.org/grapher/infant-mortality?tab=chart&country=ARG
https://ourworldindata.org/grapher/literacy-rate-adults?tab=chart&country=ARG
https://ourworldindata.org/grapher/mobile-cellular-subscriptions-by-country?tab=chart&country=ARG
https://ourworldindata.org/grapher/milk-production-tonnes?tab=chart&country=ARG
https://ourworldindata.org/grapher/meat-production-tonnes?tab=chart&country=ARG
https://ourworldindata.org/grapher/crude-death-rate-the-share-of-the-population-that-dies-per-year?ta
b=chart&country=ARG
https://ourworldindata.org/grapher/life-expectancy?tab=chart&country=ARG
https://ourworldindata.org/grapher/value-of-agricultural-production?tab=chart&country=ARG
https://ourworldindata.org/grapher/agricultural-land?tab=chart&country=ARG
https://ourworldindata.org/grapher/value-of-agricultural-production?tab=chart&country=ARG
https://ourworldindata.org/grapher/share-of-land-area-used-for-agriculture?tab=chart&country=ARG
https://ourworldindata.org/grapher/crude-birth-rate?tab=chart&country=ARGher/per-capita-milk-consu
mption?tab=chart&country=ARG
https://ourworldindata.org/graph
https://mercados.ambito.com//dolar/dolar-oficial/historico-general/
https://www.dolarsi.com/api/api.php?type=dolar
https://mercados.ambito.com//dolar/informal/historico-general/
https://api.estadisticasbcra.com/inflacion_mensual_oficial

