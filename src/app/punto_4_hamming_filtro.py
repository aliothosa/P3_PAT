"""
Punto 4 - Recorte por potencia, preénfasis y ventaneo Hamming.

Cambios importantes:
- Antes se usaba todo el audio, incluyendo silencios al inicio/final.
- Ahora se calcula un arreglo de potencias de corto tiempo para detectar dónde está
  realmente la palabra y recortar el audio antes de extraer LPC.
- También se filtran marcos con potencia muy baja después del ventaneo.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.app.punto_3_muestreo import muestreo_audios

AudioProcesado = Dict[str, object]


def aplicar_preenfasis(audio: np.ndarray, coeficiente: float = 0.95) -> np.ndarray:
    """
    Aplica filtro de preénfasis:
        y[n] = x[n] - coeficiente * x[n - 1]
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return audio.copy()

    audio_preenfatizado = np.zeros_like(audio)
    audio_preenfatizado[0] = audio[0]
    audio_preenfatizado[1:] = audio[1:] - coeficiente * audio[:-1]

    return audio_preenfatizado


def calcular_potencias(
    audio: np.ndarray,
    N: int = 160,
    M: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la potencia de corto tiempo de un audio.

    Args:
        audio: Señal de audio.
        N: Tamaño de ventana para medir potencia.
        M: Salto entre ventanas.

    Returns:
        potencias: arreglo de potencia por ventana.
        inicios: índice de muestra donde inicia cada ventana.
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=int)

    if len(audio) < N:
        audio = np.pad(audio, (0, N - len(audio)))

    potencias = []
    inicios = []

    for inicio in range(0, len(audio) - N + 1, M):
        marco = audio[inicio:inicio + N]
        potencias.append(float(np.mean(marco ** 2)))
        inicios.append(inicio)

    return np.asarray(potencias, dtype=np.float64), np.asarray(inicios, dtype=int)


def suavizar_potencias(potencias: np.ndarray, ventana: int = 5) -> np.ndarray:
    """
    Suaviza el arreglo de potencias para evitar que pequeños saltos corten la palabra.
    """
    potencias = np.asarray(potencias, dtype=np.float64)

    if len(potencias) == 0 or ventana <= 1:
        return potencias

    ventana = min(ventana, len(potencias))
    kernel = np.ones(ventana, dtype=np.float64) / ventana
    return np.convolve(potencias, kernel, mode="same")


def segmento_activo_mas_largo(mascara: np.ndarray, max_silencio: int = 3) -> Tuple[int, int]:
    """
    Devuelve el inicio y fin del segmento activo principal.

    max_silencio permite unir regiones activas separadas por silencios pequeños.
    Esto ayuda cuando una palabra tiene una pausa interna muy corta.
    """
    mascara = np.asarray(mascara, dtype=bool)

    if not np.any(mascara):
        return 0, len(mascara) - 1

    # Rellenar huecos pequeños entre regiones activas.
    mascara_unida = mascara.copy()
    indices_activos = np.where(mascara)[0]

    for i in range(len(indices_activos) - 1):
        actual = indices_activos[i]
        siguiente = indices_activos[i + 1]
        hueco = siguiente - actual - 1

        if 0 < hueco <= max_silencio:
            mascara_unida[actual:siguiente + 1] = True

    indices = np.where(mascara_unida)[0]

    # Buscar la región continua más larga.
    mejor_inicio = indices[0]
    mejor_fin = indices[0]
    inicio_actual = indices[0]
    fin_actual = indices[0]

    for idx in indices[1:]:
        if idx == fin_actual + 1:
            fin_actual = idx
        else:
            if (fin_actual - inicio_actual) > (mejor_fin - mejor_inicio):
                mejor_inicio = inicio_actual
                mejor_fin = fin_actual
            inicio_actual = idx
            fin_actual = idx

    if (fin_actual - inicio_actual) > (mejor_fin - mejor_inicio):
        mejor_inicio = inicio_actual
        mejor_fin = fin_actual

    return int(mejor_inicio), int(mejor_fin)


def recortar_por_potencia(
    audio: np.ndarray,
    N: int = 160,
    M: int = 64,
    umbral_relativo: float = 0.12,
    margen_marcos: int = 8,
    suavizado: int = 5,
    max_silencio: int = 4,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Recorta el audio tomando el segmento donde está la palabra según potencia.

    La decisión se hace con un umbral adaptativo:
        umbral = piso_ruido + umbral_relativo * (max_potencia - piso_ruido)

    donde piso_ruido se aproxima con un percentil bajo de las potencias.

    Returns:
        audio_recortado, info_recorte
    """
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

    potencias, inicios = calcular_potencias(audio, N=N, M=M)
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
    max_potencia = float(np.max(potencias_suavizadas))
    umbral = piso_ruido + umbral_relativo * (max_potencia - piso_ruido)

    mascara_activa = potencias_suavizadas >= umbral
    marco_inicio, marco_fin = segmento_activo_mas_largo(
        mascara_activa,
        max_silencio=max_silencio,
    )

    marco_inicio = max(0, marco_inicio - margen_marcos)
    marco_fin = min(len(potencias_suavizadas) - 1, marco_fin + margen_marcos)

    inicio_muestra = int(inicios[marco_inicio]) if len(inicios) else 0
    fin_muestra = int(inicios[marco_fin] + N) if len(inicios) else len(audio)
    inicio_muestra = max(0, inicio_muestra)
    fin_muestra = min(len(audio), fin_muestra)

    # Protección: si por algún motivo el segmento quedó demasiado corto,
    # conservamos el audio original.
    if fin_muestra - inicio_muestra < N:
        inicio_muestra = 0
        fin_muestra = len(audio)

    audio_recortado = audio[inicio_muestra:fin_muestra]

    info_recorte: Dict[str, object] = {
        "potencias": potencias,
        "potencias_suavizadas": potencias_suavizadas,
        "umbral_potencia": umbral,
        "piso_ruido": piso_ruido,
        "max_potencia": max_potencia,
        "recorte_inicio": inicio_muestra,
        "recorte_fin": fin_muestra,
        "marco_inicio": marco_inicio,
        "marco_fin": marco_fin,
        "duracion_original_muestras": len(audio),
        "duracion_recortada_muestras": len(audio_recortado),
    }

    return audio_recortado, info_recorte


def aplicar_ventaneo(
    audio_preenfatizado: np.ndarray,
    N: int = 160,
    M: int = 64,
) -> np.ndarray:
    """
    Divide el audio en marcos de N muestras con salto M y aplica ventana Hamming.
    """
    audio_preenfatizado = np.asarray(audio_preenfatizado, dtype=np.float64)

    if N <= 0 or M <= 0:
        raise ValueError("N y M deben ser mayores que cero.")

    if len(audio_preenfatizado) < N:
        audio_preenfatizado = np.pad(audio_preenfatizado, (0, N - len(audio_preenfatizado)))

    n = np.arange(N)
    ventana_hamming = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))

    marcos_listos = []

    for inicio in range(0, len(audio_preenfatizado) - N + 1, M):
        marco = audio_preenfatizado[inicio:inicio + N]
        marcos_listos.append(marco * ventana_hamming)

    return np.asarray(marcos_listos, dtype=np.float64)


def filtrar_marcos_por_potencia(
    marcos: np.ndarray,
    umbral_relativo: float = 0.04,
    min_marcos: int = 8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Elimina marcos de muy baja potencia después del recorte.

    Esto evita que los silencios residuales entren al LPC/cuantizador.
    """
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

    if np.sum(mascara) < min_marcos:
        cantidad = min(min_marcos, len(marcos))
        indices_top = np.argsort(potencias)[-cantidad:]
        mascara = np.zeros(len(marcos), dtype=bool)
        mascara[indices_top] = True

    marcos_filtrados = marcos[mascara]

    info = {
        "potencias_marcos": potencias,
        "umbral_marcos": umbral,
        "mascara_marcos": mascara,
        "marcos_descartados": int(len(marcos) - len(marcos_filtrados)),
    }

    return marcos_filtrados, info


def procesar_audios(
    audios: Optional[Dict[str, List[dict]]] = None,
    coeficiente_preenfasis: float = 0.95,
    N: int = 160,
    M: int = 64,
    usar_recorte_por_potencia: bool = True,
    umbral_recorte: float = 0.12,
    margen_marcos: int = 8,
    filtrar_marcos_bajos: bool = True,
    umbral_marcos: float = 0.04,
) -> Dict[str, List[AudioProcesado]]:
    """
    Aplica recorte por potencia, preénfasis, ventaneo y filtrado de marcos.
    """
    if audios is None:
        audios = muestreo_audios()

    senales_procesadas: Dict[str, List[AudioProcesado]] = {}

    for etiqueta, lista_audios in audios.items():
        senales_procesadas[etiqueta] = []

        for audio_info in lista_audios:
            audio = np.asarray(audio_info["senal"], dtype=np.float64)

            if usar_recorte_por_potencia:
                audio_util, info_recorte = recortar_por_potencia(
                    audio,
                    N=N,
                    M=M,
                    umbral_relativo=umbral_recorte,
                    margen_marcos=margen_marcos,
                )
            else:
                audio_util = audio
                potencias, _ = calcular_potencias(audio, N=N, M=M)
                info_recorte = {
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

            audio_pe = aplicar_preenfasis(audio_util, coeficiente=coeficiente_preenfasis)
            marcos_originales = aplicar_ventaneo(audio_pe, N=N, M=M)

            if filtrar_marcos_bajos:
                marcos, info_filtro = filtrar_marcos_por_potencia(
                    marcos_originales,
                    umbral_relativo=umbral_marcos,
                )
            else:
                marcos = marcos_originales
                info_filtro = {
                    "potencias_marcos": np.mean(marcos_originales ** 2, axis=1)
                    if len(marcos_originales) else np.asarray([], dtype=np.float64),
                    "umbral_marcos": 0.0,
                    "mascara_marcos": np.ones(len(marcos_originales), dtype=bool),
                    "marcos_descartados": 0,
                }

            senales_procesadas[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": audio_info["archivo"],
                    "ruta": audio_info["ruta"],
                    "sr": audio_info["sr"],
                    "senal_original": audio,
                    "senal_recortada": audio_util,
                    "preenfatizado": audio_pe,
                    "marcos": marcos,
                    "num_marcos": len(marcos),
                    "num_marcos_antes_filtro": len(marcos_originales),
                    **info_recorte,
                    **info_filtro,
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
                f"  {audio['archivo']}: marcos {audio['num_marcos_antes_filtro']} -> {audio['num_marcos']} | "
                f"recorte {audio['recorte_inicio']}:{audio['recorte_fin']}"
            )
