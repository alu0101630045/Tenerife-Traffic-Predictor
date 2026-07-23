import pandas as pd
import pydeck as pdk
import streamlit as st

from api_estaciones import cargar_estaciones_aforo
from modelo import entrenar_modelo
from api_trafico import cargar_datos_api
from preprocesamiento import (
    estimar_posibilidad_atasco,
    preprocesar_datos,
    preparar_entrada_usuario,
)


URL_BASE = "https://datos.tenerife.es/ckan"
ID_PAQUETE = "d02ccf54-aca2-4a82-9ef0-fbd285b5f401"
ID_PAQUETE_ESTACIONES = "a114f82b-bb62-47e0-94bb-bebb484514a7"
CACHE_VERSION = "v3-map-station-id"


st.set_page_config(page_title="Predictor de Tráfico", layout="centered")


st.title("Predictor de Tráfico")
st.write("Selecciona un tramo, un sentido y una fecha para estimar el volumen de vehículos.")


@st.cache_data(show_spinner="Descargando y procesando datos...")
def cargar_datos_preparados(cache_version=CACHE_VERSION):
    dataset_crudo = cargar_datos_api(URL_BASE, ID_PAQUETE)
    return preprocesar_datos(dataset_crudo)


@st.cache_resource(show_spinner="Entrenando modelo...")
def obtener_modelo(X, y, cache_version=CACHE_VERSION):
    return entrenar_modelo(X, y)


@st.cache_data(show_spinner="Cargando estaciones del mapa...")
def cargar_estaciones_geograficas(cache_version=CACHE_VERSION):
    estaciones = cargar_estaciones_aforo(URL_BASE, ID_PAQUETE_ESTACIONES)
    columnas = [
        "estacion_id",
        "estacion_nombre",
        "carretera_codigo",
        "carretera_nombre",
        "tramo_nombre",
        "tramo_orden",
        "estacion_pk",
        "estacion_latitud",
        "estacion_longitud",
    ]
    estaciones = estaciones[columnas].copy()
    estaciones = estaciones.dropna(subset=["estacion_latitud", "estacion_longitud"])
    estaciones["estacion_id"] = estaciones["estacion_id"].astype(int)
    estaciones["etiqueta"] = estaciones.apply(
        lambda fila: f"{fila['estacion_nombre']} · {fila['carretera_codigo']} / {fila['tramo_nombre']}",
        axis=1,
    )
    return estaciones


def crear_mapa_estaciones(estaciones, estacion_id_seleccionada):
    estaciones = estaciones.copy()
    estacion_seleccionada = estaciones[estaciones["estacion_id"] == estacion_id_seleccionada]

    if estacion_seleccionada.empty:
        raise ValueError("No se encontró la estación seleccionada en el mapa")

    estacion_actual = estacion_seleccionada.iloc[0]
    estaciones_otras = estaciones[estaciones["estacion_id"] != estacion_id_seleccionada]

    capa_otras = pdk.Layer(
        "ScatterplotLayer",
        data=estaciones_otras,
        get_position="[estacion_longitud, estacion_latitud]",
        get_fill_color=[90, 125, 180, 90],
        get_line_color=[255, 255, 255, 120],
        get_radius=5,
        radius_units="pixels",
        radius_min_pixels=4,
        stroked=True,
        pickable=True,
    )

    capa_seleccionada = pdk.Layer(
        "ScatterplotLayer",
        data=estacion_seleccionada,
        get_position="[estacion_longitud, estacion_latitud]",
        get_fill_color=[220, 53, 69, 220],
        get_line_color=[255, 255, 255, 255],
        get_radius=12,
        radius_units="pixels",
        radius_min_pixels=10,
        stroked=True,
        pickable=True,
    )

    capa_texto = pdk.Layer(
        "TextLayer",
        data=estacion_seleccionada,
        get_position="[estacion_longitud, estacion_latitud]",
        get_text="etiqueta",
        get_size=14,
        get_color=[30, 30, 30],
        get_text_anchor="middle",
        get_alignment_baseline="bottom",
    )

    vista = pdk.ViewState(
        latitude=float(estacion_actual["estacion_latitud"]),
        longitude=float(estacion_actual["estacion_longitud"]),
        zoom=9.3,
        pitch=0,
    )

    return pdk.Deck(
        layers=[capa_otras, capa_seleccionada, capa_texto],
        initial_view_state=vista,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "text": "{etiqueta}\nCarretera: {carretera_codigo}\nTramo: {tramo_nombre}",
        },
    )


def main():
    try:
        X, y, le_estacion, df_agrupado = cargar_datos_preparados()
        modelo_entrenado = obtener_modelo(X, y)
        estaciones_geo = cargar_estaciones_geograficas()

        if "estacion_id" not in df_agrupado.columns:
            mapa_nombre_id = (
                estaciones_geo.drop_duplicates("estacion_nombre")
                .set_index("estacion_nombre")["estacion_id"]
                .to_dict()
            )
            df_agrupado = df_agrupado.copy()
            df_agrupado["estacion_id"] = df_agrupado["estacion_nombre"].map(mapa_nombre_id)
            if df_agrupado["estacion_id"].isna().any():
                estaciones_faltantes = sorted(df_agrupado.loc[df_agrupado["estacion_id"].isna(), "estacion_nombre"].unique())
                raise KeyError(f"No se pudieron resolver estas estaciones: {estaciones_faltantes}")
            df_agrupado["estacion_id"] = df_agrupado["estacion_id"].astype(int)

        estaciones_disponibles = estaciones_geo[
            estaciones_geo["estacion_id"].isin(df_agrupado["estacion_id"].unique())
        ].sort_values(["carretera_codigo", "tramo_orden", "estacion_nombre"])

        etiquetas_estaciones = estaciones_disponibles.set_index("estacion_id")["etiqueta"].to_dict()
        opciones_estacion = estaciones_disponibles["estacion_id"].tolist()

        st.subheader("Mapa de estaciones")
        estacion_id = st.selectbox(
            "Estación",
            opciones_estacion,
            format_func=lambda valor: etiquetas_estaciones[valor],
        )
        estacion_actual = estaciones_disponibles[estaciones_disponibles["estacion_id"] == estacion_id].iloc[0]

        col_mapa, col_info = st.columns([2.2, 1])
        with col_mapa:
            st.pydeck_chart(crear_mapa_estaciones(estaciones_disponibles, estacion_id), use_container_width=True)
        with col_info:
            st.metric("Estación seleccionada", estacion_actual["estacion_nombre"])
            st.metric("Carretera", estacion_actual["carretera_codigo"])
            st.write(f"**Tramo:** {estacion_actual['tramo_nombre']}")
            st.write(f"**PK:** {estacion_actual['estacion_pk']:.2f}")
            st.caption("Los puntos grises son las estaciones disponibles y el rojo marca la estación elegida.")

        col1, col2 = st.columns(2)
        col1.metric("Registros", f"{len(df_agrupado):,}")
        col2.metric("Estaciones", f"{df_agrupado['estacion_id'].nunique():,}")

        with st.form("form_prediccion"):
            sentidos_disponibles = sorted(df_agrupado[df_agrupado["estacion_id"] == estacion_id]["sentido"].unique())
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

            entrada = preparar_entrada_usuario(estacion_id, sentido, fecha_str, le_estacion)
            prediccion = modelo_entrenado.predict(entrada)[0]
            hay_atasco, nivel_riesgo, umbral_posible, umbral_alto = estimar_posibilidad_atasco(
                df_agrupado,
                estacion_id,
                sentido,
                fecha_hora,
                prediccion,
            )

            texto_sentido = "Creciente" if sentido == 1 else ("Decreciente" if sentido == -1 else "Único")
            texto_atasco = "Sí" if hay_atasco else "No"
            texto_riesgo = {
                "bajo": "Bajo",
                "medio": "Medio",
                "alto": "Alto",
            }.get(nivel_riesgo, nivel_riesgo.title())
            etiqueta_estacion = etiquetas_estaciones[estacion_id]

            st.success("Predicción generada correctamente")
            st.subheader("Resultado")
            st.write(f"**Estación:** {etiqueta_estacion}")
            st.write(f"**Sentido:** {texto_sentido}")
            st.write(f"**Fecha:** {fecha_str}")
            st.metric("Volumen estimado", f"{prediccion:.0f} vehículos/hora")
            st.metric("Posibilidad de atasco", texto_atasco)
            st.caption(f"Nivel de riesgo: {texto_riesgo}")
            st.caption(
                f"Umbral posible: {umbral_posible:.0f} vehículos/hora | Umbral alto: {umbral_alto:.0f} vehículos/hora"
            )

            if hay_atasco:
                st.warning("La predicción entra en una zona con riesgo de atasco según el histórico.")
            else:
                st.info("La predicción se mantiene dentro del comportamiento histórico habitual.")

    except Exception as e:
        st.error(f"No se pudo cargar el panel: {e}")


if __name__ == "__main__":
    main()