# Tenerife Traffic Predictor

Aplicación web en Streamlit para estimar el volumen de tráfico en una estación concreta de Tenerife, mostrar una posible señal de atasco y ubicar la estación en un mapa interactivo.

La app consume datos públicos del Cabildo de Tenerife y trabaja con dos fuentes distintas:

- el dataset de intensidad diaria de tráfico;
- el catálogo geográfico de estaciones de aforo.

## Qué hace el proyecto

- Carga datos de tráfico desde la API pública de Tenerife.
- Prepara las variables temporales y de tramo para el modelo.
- Entrena un `RandomForestRegressor` sobre los datos históricos.
- Muestra un mapa de la isla con la estación seleccionada resaltada.
- Calcula una estimación de posibilidad de atasco a partir del histórico.

## Estructura del repositorio

- [src/app.py](src/app.py): interfaz principal de Streamlit.
- [src/api_trafico.py](src/api_trafico.py): descarga del dataset de tráfico.
- [src/api_estaciones.py](src/api_estaciones.py): descarga del catálogo geográfico de estaciones.
- [src/preprocesamiento.py](src/preprocesamiento.py): limpieza de datos y preparación de entrada.
- [src/modelo.py](src/modelo.py): entrenamiento del modelo de predicción.
- [streamlit_app.py](streamlit_app.py): punto de entrada para Streamlit Community Cloud.
- [requirements.txt](requirements.txt): dependencias del proyecto.
- [runtime.txt](runtime.txt): versión de Python usada en despliegue.
- [.streamlit/config.toml](.streamlit/config.toml): configuración de Streamlit.
- [notebooks/predictor_trafico_documentado.ipynb](notebooks/predictor_trafico_documentado.ipynb): cuaderno documentado del proyecto.

## Requisitos

Para ejecutar el proyecto en local necesitas:

- Python 3.12.3 o compatible;
- acceso a Internet para consultar la API de Tenerife;
- las dependencias del fichero `requirements.txt`.

## Instalación local

1. Clona el repositorio.
2. Crea y activa un entorno virtual.
3. Instala las dependencias.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la aplicación

La forma más cómoda de arrancarla en local es con Streamlit usando el archivo principal de la carpeta `src`:

```bash
python -m streamlit run src/app.py
```

También puedes usar el entrypoint preparado para la nube:

```bash
python -m streamlit run streamlit_app.py
```

## Cómo funciona la aplicación

1. Se descargan los datos de tráfico y las estaciones de aforo.
2. Se agrupan los registros por estación, sentido, fecha y hora.
3. Se entrena el modelo con el histórico disponible.
4. El usuario elige una estación, un sentido, una fecha y una hora.
5. La app devuelve el volumen estimado, una señal de riesgo de atasco y la estación aparece marcada en el mapa.

## Ejemplos de código

### Cargar datos de tráfico

```python
from api_trafico import cargar_datos_api

URL_BASE = "https://datos.tenerife.es/ckan"
ID_PAQUETE = "d02ccf54-aca2-4a82-9ef0-fbd285b5f401"

df = cargar_datos_api(URL_BASE, ID_PAQUETE)
```

### Preparar el dataset

```python
from preprocesamiento import preprocesar_datos

X, y, le_estacion, df_agrupado = preprocesar_datos(df)
```

### Entrenar el modelo

```python
from modelo import entrenar_modelo

modelo = entrenar_modelo(X, y)
```

### Preparar una predicción

```python
from preprocesamiento import preparar_entrada_usuario

entrada = preparar_entrada_usuario(
    estacion_id=11,
    sentido=-1,
    fecha_str="2026-06-01 08:00",
    le_estacion=le_estacion,
)

prediccion = modelo.predict(entrada)[0]
```

### Estimar posibilidad de atasco

```python
from preprocesamiento import estimar_posibilidad_atasco

hay_atasco, nivel_riesgo, umbral_posible, umbral_alto = estimar_posibilidad_atasco(
    df_agrupado,
    estacion=11,
    sentido=-1,
    fecha_hora="2026-06-01 08:00",
    prediccion=prediccion,
)
```

## Despliegue en Streamlit Community Cloud

El proyecto ya está preparado para desplegarse ahí.

1. Sube los cambios a GitHub.
2. En Streamlit Community Cloud, crea o actualiza la app.
3. Selecciona `streamlit_app.py` como archivo principal.
4. Deja que la plataforma instale las dependencias y arranque la app.

Archivos clave para el despliegue:

- [streamlit_app.py](streamlit_app.py)
- [requirements.txt](requirements.txt)
- [runtime.txt](runtime.txt)
- [.streamlit/config.toml](.streamlit/config.toml)

## Notas útiles

- El proyecto depende de la API pública del Cabildo de Tenerife, así que si el servicio externo falla, la app puede tardar en cargar o mostrar error.
- El mapa usa coordenadas oficiales del catálogo de estaciones.
- Los ficheros de datos locales no forman parte del repositorio; todo se descarga en tiempo de ejecución.

## Licencia

No se ha definido una licencia específica en el repositorio.