import requests
import pandas as pd


def cargar_datos_api(url, id_paquete):
  endpoint = f"{url}/api/3/action/package_show"
  respuesta = requests.get(endpoint, params={"id": id_paquete}, timeout=20)

  if respuesta.status_code != 200:
    try:
      detalle_error = respuesta.json().get("error", {})
    except ValueError:
      detalle_error = respuesta.text
    raise ConnectionError(
      f"Error en la petición. Código: {respuesta.status_code}. Detalle: {detalle_error}"
    )

  datos_json = respuesta.json()
  recursos = datos_json.get('result', {}).get('resources', [])
  url_descarga = None

  # Dado que el paquete puede tener recursos en JSON y TXT, buscamos solo el recurso TXT que viene en formato CSV.
  for recurso in recursos:
    if recurso.get('format', '').upper() == 'TXT':
      url_descarga = recurso.get('url')
      break
  
  if not url_descarga:
    raise ValueError("No se encontró ningún fichero de datos TXT")

  df = pd.read_csv(url_descarga)

  return df