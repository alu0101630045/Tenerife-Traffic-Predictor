import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocesar_datos(df):
    # Formateamos fecha y juntamos todos los vehículos en una sola columna
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['total_vehiculos'] = df['ligeros'] + df['pesados']
    
    # La variable sentido representa de manera númerica si es creciente, decreciente o tiene un único sentido.
    df['sentido'] = df['carril_id'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Sumamos los vehículos de todos los carriles de un mismo sentido para una determinada fecha y tramo.
    df_agrupado = df.groupby(['fecha', 'estacion_nombre', 'sentido'], as_index=False)['total_vehiculos'].sum()
    
    # Formateamos hora y día de la semana para que el modelo detecte patrones temporales. Esto nos ayudará a que el modelo
    # aprenda que a las 4 de la madrugada suele haber mucho menos tráfico que a las 4 de la tarde.
    df_agrupado['hora'] = df_agrupado['fecha'].dt.hour
    df_agrupado['dia_semana'] = df_agrupado['fecha'].dt.dayofweek
    
    # Codificamos las estaciones en enteros para que el modelo pueda procesarlas.
    le_estacion = LabelEncoder()
    df_agrupado['estacion_encoded'] = le_estacion.fit_transform(df_agrupado['estacion_nombre'])
    
    # La entrada al modelo será un DataFrame con la estación, sentido, hora y día de la semana, y nos devolverá el total de vehículos 
    # que se esperan en ese tramo, sentido y hora.
    features = ['estacion_encoded', 'sentido', 'hora', 'dia_semana']
    X = df_agrupado[features]
    y = df_agrupado['total_vehiculos']
    
    return X, y, le_estacion, df_agrupado

def preparar_entrada_usuario(estacion, sentido, fecha_str, le_estacion):
    # Formateamos la fecha y extraemos la hora y el día de la semana para que el modelo pueda hacer la predicción.
    fecha = pd.to_datetime(fecha_str)
    hora = fecha.hour
    dia_semana = fecha.dayofweek
    # Volvemos a codificar la estación que el usuario ha pedido.
    estacion_encoded = le_estacion.transform([estacion])[0]
    
    # Creamos un DataFrame con todos los campos de entrada que el modelo espera.
    entrada = pd.DataFrame({
        'estacion_encoded': [estacion_encoded],
        'sentido': [sentido],
        'hora': [hora],
        'dia_semana': [dia_semana]
    })
    
    return entrada