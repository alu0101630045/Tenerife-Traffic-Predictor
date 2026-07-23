import pandas as pd
import streamlit as st

from api_carga import cargar_datos_api
from modelo import entrenar_modelo
from preprocesamiento import preprocesar_datos, preparar_entrada_usuario


URL_BASE = "https://datos.tenerife.es/ckan"
ID_PAQUETE = "d02ccf54-aca2-4a82-9ef0-fbd285b5f401"


st.set_page_config(page_title="Predictor de Tráfico", layout="centered")


st.title("Predictor de Tráfico")
st.write("Selecciona un tramo, un sentido y una fecha para estimar el volumen de vehículos.")


@st.cache_data(show_spinner="Descargando y procesando datos...")
def cargar_datos_preparados():
    dataset_crudo = cargar_datos_api(URL_BASE, ID_PAQUETE)
    return preprocesar_datos(dataset_crudo)


@st.cache_resource(show_spinner="Entrenando modelo...")
def obtener_modelo(X, y):
    return entrenar_modelo(X, y)


try:
    X, y, le_estacion, df_agrupado = cargar_datos_preparados()
    modelo_entrenado = obtener_modelo(X, y)

    col1, col2 = st.columns(2)
    col1.metric("Registros", f"{len(df_agrupado):,}")
    col2.metric("Tramos", f"{df_agrupado['estacion_nombre'].nunique():,}")

    with st.form("form_prediccion"):
        estaciones = sorted(df_agrupado["estacion_nombre"].unique())
        estacion = st.selectbox("Tramo", estaciones)

        sentidos_disponibles = sorted(
            df_agrupado[df_agrupado["estacion_nombre"] == estacion]["sentido"].unique()
        )
        sentido = st.selectbox(
            "Sentido",
            sentidos_disponibles,
            format_func=lambda valor: {
                1: "Creciente (Ej: Hacia Santa Cruz)",
                -1: "Decreciente (Ej: Hacia el Sur)",
                0: "Único sentido (Ej: Calle de un solo sentido)",
            }.get(valor, str(valor)),
        )

        fecha = st.date_input("Fecha")
        hora = st.time_input("Hora")

        enviar = st.form_submit_button("Predecir")

    if enviar:
        fecha_hora = pd.Timestamp.combine(fecha, hora)
        fecha_str = fecha_hora.strftime("%Y-%m-%d %H:%M")

        entrada = preparar_entrada_usuario(estacion, sentido, fecha_str, le_estacion)
        prediccion = modelo_entrenado.predict(entrada)[0]

        texto_sentido = "Creciente" if sentido == 1 else ("Decreciente" if sentido == -1 else "Único")

        st.success("Predicción generada correctamente")
        st.subheader("Resultado")
        st.write(f"**Tramo:** {estacion}")
        st.write(f"**Sentido:** {texto_sentido}")
        st.write(f"**Fecha:** {fecha_str}")
        st.metric("Volumen estimado", f"{prediccion:.0f} vehículos/hora")

except Exception as e:
    st.error(f"No se pudo cargar el panel: {e}")