from typing import Dict, List, Optional, Tuple

import numpy as np

from src.app.punto_3_muestreo import cargar_audios

AudioProcesado = Dict[str, object]



# Aplica filtro de preénfasis: y[n] = x[n] - coeficiente * x[n - 1]
def aplicar_preenfasis(audio: np.ndarray, coeficiente: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return audio.copy()

    audio_preenfatizado = np.zeros_like(audio)
    audio_preenfatizado[0] = audio[0]
    audio_preenfatizado[1:] = audio[1:] - coeficiente * audio[:-1]

    return audio_preenfatizado


def calcular_potencias(audio: np.ndarray, tamano_ventana: int = 160, salto_ventana: int = 64) -> Tuple[np.ndarray, np.ndarray]:

    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=int)

    if len(audio) < tamano_ventana:
        audio = np.pad(audio, (0, tamano_ventana - len(audio)))

    potencias = []
    inicios = []

    for inicio in range(0, len(audio) - tamano_ventana + 1, salto_ventana):
        marco = audio[inicio:inicio + tamano_ventana]
        potencias.append(float(np.mean(marco ** 2)))
        inicios.append(inicio)

    return np.asarray(potencias, dtype=np.float64), np.asarray(inicios, dtype=int)

# Suaviza el arreglo de potencias para evitar que pequeños saltos corten la palabra.
def suavizar_potencias(potencias: np.ndarray, ventana: int = 5) -> np.ndarray:
    potencias = np.asarray(potencias, dtype=np.float64)

    if len(potencias) == 0 or ventana <= 1:
        return potencias

    ventana = min(ventana, len(potencias))
    nucleo = np.ones(ventana, dtype=np.float64) / ventana
    return np.convolve(potencias, nucleo, mode="same")

# Devuelve el inicio y fin del segmento activo principal.
def segmento_activo_mas_largo(mascara: np.ndarray, maximo_silencio: int = 3) -> Tuple[int, int]:
    mascara = np.asarray(mascara, dtype=bool)

    if not np.any(mascara):
        return 0, len(mascara) - 1

    # Rellenar huecos pequeños entre regiones activas.
    mascara_unida = mascara.copy()
    indices_activos = np.where(mascara)[0]

    for indice in range(len(indices_activos) - 1):
        actual = indices_activos[indice]
        siguiente = indices_activos[indice + 1]
        hueco = siguiente - actual - 1

        if 0 < hueco <= maximo_silencio:
            mascara_unida[actual:siguiente + 1] = True

    indices = np.where(mascara_unida)[0]

    # Buscar la región continua más larga.
    mejor_inicio = indices[0]
    mejor_fin = indices[0]
    inicio_actual = indices[0]
    fin_actual = indices[0]

    for indice in indices[1:]:
        if indice == fin_actual + 1:
            fin_actual = indice
        else:
            if (fin_actual - inicio_actual) > (mejor_fin - mejor_inicio):
                mejor_inicio = inicio_actual
                mejor_fin = fin_actual
            inicio_actual = indice
            fin_actual = indice

    if (fin_actual - inicio_actual) > (mejor_fin - mejor_inicio):
        mejor_inicio = inicio_actual
        mejor_fin = fin_actual

    return int(mejor_inicio), int(mejor_fin)

"""
Recorta el audio tomando el segmento donde está la palabra según potencia.

La decisión se hace con un umbral adaptativo:
        umbral = piso_ruido + umbral_relativo * (potencia_maxima - piso_ruido)

donde piso_ruido se aproxima con un percentil bajo de las potencias.

"""

def recortar_por_potencia(audio: np.ndarray, tamano_ventana: int = 160, salto_ventana: int = 64, umbral_relativo: float = 0.12, margen_marcos: int = 8, suavizado: int = 5, maximo_silencio: int = 4) -> Tuple[np.ndarray, Dict[str, object]]:

    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return audio.copy(), {
            "potencias": np.asarray([], dtype=np.float64),
            "potencias_suavizadas": np.asarray([], dtype=np.float64),
            "umbral_potencia": 0.0,
            "recorte_inicio": 0,
            "recorte_fin": 0,
            "marco_inicio": 0,
            "marco_fin": 0,
        }

    potencias, inicios = calcular_potencias(audio, tamano_ventana=tamano_ventana, salto_ventana=salto_ventana)
    potencias_suavizadas = suavizar_potencias(potencias, ventana=suavizado)

    if len(potencias_suavizadas) == 0 or float(np.max(potencias_suavizadas)) <= 1e-14:
        return audio.copy(), {
            "potencias": potencias,
            "potencias_suavizadas": potencias_suavizadas,
            "umbral_potencia": 0.0,
            "recorte_inicio": 0,
            "recorte_fin": len(audio),
            "marco_inicio": 0,
            "marco_fin": max(0, len(potencias) - 1),
        }

    piso_ruido = float(np.percentile(potencias_suavizadas, 15))
    potencia_maxima = float(np.max(potencias_suavizadas))
    umbral = piso_ruido + umbral_relativo * (potencia_maxima - piso_ruido)

    mascara_activa = potencias_suavizadas >= umbral
    marco_inicio, marco_fin = segmento_activo_mas_largo(mascara_activa, maximo_silencio=maximo_silencio)

    marco_inicio = max(0, marco_inicio - margen_marcos)
    marco_fin = min(len(potencias_suavizadas) - 1, marco_fin + margen_marcos)

    inicio_muestra = int(inicios[marco_inicio]) if len(inicios) else 0
    fin_muestra = int(inicios[marco_fin] + tamano_ventana) if len(inicios) else len(audio)
    inicio_muestra = max(0, inicio_muestra)
    fin_muestra = min(len(audio), fin_muestra)

    # Protección: si por algún motivo el segmento quedó demasiado corto,
    # conservamos el audio original.
    if fin_muestra - inicio_muestra < tamano_ventana:
        inicio_muestra = 0
        fin_muestra = len(audio)

    audio_recortado = audio[inicio_muestra:fin_muestra]

    informacion_recorte: Dict[str, object] = {
        "potencias": potencias,
        "potencias_suavizadas": potencias_suavizadas,
        "umbral_potencia": umbral,
        "piso_ruido": piso_ruido,
        "potencia_maxima": potencia_maxima,
        "recorte_inicio": inicio_muestra,
        "recorte_fin": fin_muestra,
        "marco_inicio": marco_inicio,
        "marco_fin": marco_fin,
        "duracion_original_muestras": len(audio),
        "duracion_recortada_muestras": len(audio_recortado),
    }

    return audio_recortado, informacion_recorte

# Divide el audio en marcos de "tamano_ventana" muestras con "salto_ventana" y aplica ventana Hamming.
def aplicar_ventaneo(audio_preenfatizado: np.ndarray, tamano_ventana: int = 160, salto_ventana: int = 64) -> np.ndarray:
 
    audio_preenfatizado = np.asarray(audio_preenfatizado, dtype=np.float64)

    if tamano_ventana <= 0 or salto_ventana <= 0:
        raise ValueError("tamano_ventana y salto_ventana deben ser mayores que cero.")

    if len(audio_preenfatizado) < tamano_ventana:
        audio_preenfatizado = np.pad(audio_preenfatizado, (0, tamano_ventana - len(audio_preenfatizado)))

    indices_ventana = np.arange(tamano_ventana)
    ventana_hamming = 0.54 - 0.46 * np.cos((2 * np.pi * indices_ventana) / (tamano_ventana - 1))

    marcos_ventaneados = []

    for inicio in range(0, len(audio_preenfatizado) - tamano_ventana + 1, salto_ventana):
        marco = audio_preenfatizado[inicio:inicio + tamano_ventana]
        marcos_ventaneados.append(marco * ventana_hamming)

    return np.asarray(marcos_ventaneados, dtype=np.float64)

"""
  Elimina marcos de muy baja potencia después del recorte.
  Esto evita que los silencios residuales entren al LPC/cuantizador.
"""
def filtrar_marcos_por_potencia(marcos: np.ndarray, umbral_relativo: float = 0.04, minimo_marcos: int = 8) -> Tuple[np.ndarray, Dict[str, object]]:

    marcos = np.asarray(marcos, dtype=np.float64)

    if len(marcos) == 0:
        return marcos, {
            "potencias_marcos": np.asarray([], dtype=np.float64),
            "umbral_marcos": 0.0,
            "mascara_marcos": np.asarray([], dtype=bool),
            "marcos_descartados": 0,
        }

    potencias = np.mean(marcos ** 2, axis=1)

    if float(np.max(potencias)) <= 1e-14:
        mascara = np.ones(len(marcos), dtype=bool)
        return marcos, {
            "potencias_marcos": potencias,
            "umbral_marcos": 0.0,
            "mascara_marcos": mascara,
            "marcos_descartados": 0,
        }

    umbral = float(umbral_relativo * np.max(potencias))
    mascara = potencias >= umbral

    if np.sum(mascara) < minimo_marcos:
        cantidad = min(minimo_marcos, len(marcos))
        indices_mayor_potencia = np.argsort(potencias)[-cantidad:]
        mascara = np.zeros(len(marcos), dtype=bool)
        mascara[indices_mayor_potencia] = True

    marcos_filtrados = marcos[mascara]

    informacion_filtro = {
        "potencias_marcos": potencias,
        "umbral_marcos": umbral,
        "mascara_marcos": mascara,
        "marcos_descartados": int(len(marcos) - len(marcos_filtrados)),
    }

    return marcos_filtrados, informacion_filtro

# Aplica recorte por potencia, preénfasis, ventaneo y filtrado de marcos.
def procesar_audios(audios: Optional[Dict[str, List[dict]]] = None, coeficiente_preenfasis: float = 0.95, tamano_ventana: int = 160, salto_ventana: int = 64, usar_recorte_por_potencia: bool = True, umbral_recorte: float = 0.12, margen_marcos: int = 8, filtrar_marcos_bajos: bool = True, umbral_marcos: float = 0.04) -> Dict[str, List[AudioProcesado]]:

    if audios is None:
        audios = cargar_audios()

    senales_procesadas: Dict[str, List[AudioProcesado]] = {}

    for etiqueta, lista_audios in audios.items():
        senales_procesadas[etiqueta] = []

        for informacion_audio in lista_audios:
            audio = np.asarray(informacion_audio["senal"], dtype=np.float64)

            if usar_recorte_por_potencia:
                audio_util, informacion_recorte = recortar_por_potencia(
                    audio,
                    tamano_ventana=tamano_ventana,
                    salto_ventana=salto_ventana,
                    umbral_relativo=umbral_recorte,
                    margen_marcos=margen_marcos,
                )
            else:
                audio_util = audio
                potencias, _ = calcular_potencias(audio, tamano_ventana=tamano_ventana, salto_ventana=salto_ventana)
                informacion_recorte = {
                    "potencias": potencias,
                    "potencias_suavizadas": potencias,
                    "umbral_potencia": 0.0,
                    "recorte_inicio": 0,
                    "recorte_fin": len(audio),
                    "marco_inicio": 0,
                    "marco_fin": max(0, len(potencias) - 1),
                    "duracion_original_muestras": len(audio),
                    "duracion_recortada_muestras": len(audio),
                }

            audio_preenfatizado = aplicar_preenfasis(audio_util, coeficiente=coeficiente_preenfasis)
            marcos_originales = aplicar_ventaneo(audio_preenfatizado, tamano_ventana=tamano_ventana, salto_ventana=salto_ventana)

            if filtrar_marcos_bajos:
                marcos, informacion_filtro = filtrar_marcos_por_potencia(marcos_originales, umbral_relativo=umbral_marcos)
            else:
                marcos = marcos_originales
                informacion_filtro = {
                    "potencias_marcos": np.mean(marcos_originales ** 2, axis=1) if len(marcos_originales) else np.asarray([], dtype=np.float64),
                    "umbral_marcos": 0.0,
                    "mascara_marcos": np.ones(len(marcos_originales), dtype=bool),
                    "marcos_descartados": 0,
                }

            senales_procesadas[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": informacion_audio["archivo"],
                    "ruta": informacion_audio["ruta"],
                    "frecuencia_muestreo": informacion_audio["frecuencia_muestreo"],
                    "senal_original": audio,
                    "senal_recortada": audio_util,
                    "preenfatizado": audio_preenfatizado,
                    "marcos": marcos,
                    "numero_marcos": len(marcos),
                    "numero_marcos_antes_filtro": len(marcos_originales),
                    **informacion_recorte,
                    **informacion_filtro,
                }
            )

    return senales_procesadas


if __name__ == "__main__":
    senales_procesadas = procesar_audios()
    print("=== Punto 4: Audios procesados ===")
    for etiqueta, lista_audios in senales_procesadas.items():
        print(f"Etiqueta {etiqueta}: {len(lista_audios)} audios")
        for audio in lista_audios[:3]:
            print(
                f"  {audio['archivo']}: marcos {audio['numero_marcos_antes_filtro']} -> {audio['numero_marcos']} | "
                f"recorte {audio['recorte_inicio']}:{audio['recorte_fin']}"
            )
