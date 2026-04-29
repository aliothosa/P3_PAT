"""
Punto 4 - Preénfasis y ventaneo Hamming.

Rediseño:
- No procesa automáticamente al importar el archivo.
- Mantiene metadatos del audio para poder evaluar después por archivo completo.
"""

from typing import Dict, List, Optional

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


def aplicar_ventaneo(
    audio_preenfatizado: np.ndarray,
    N: int = 160,
    M: int = 64,
) -> np.ndarray:
    """
    Divide el audio en marcos de N muestras con salto M y aplica ventana Hamming.

    Args:
        audio_preenfatizado: Señal ya preenfatizada.
        N: Tamaño del marco.
        M: Salto entre marcos.

    Returns:
        Matriz de marcos con forma (num_marcos, N).
    """
    audio_preenfatizado = np.asarray(audio_preenfatizado, dtype=np.float64)

    if N <= 0 or M <= 0:
        raise ValueError("N y M deben ser mayores que cero.")

    # Si un audio es más corto que un marco, lo rellenamos con ceros para no perderlo.
    if len(audio_preenfatizado) < N:
        audio_preenfatizado = np.pad(audio_preenfatizado, (0, N - len(audio_preenfatizado)))

    n = np.arange(N)
    ventana_hamming = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))

    marcos_listos = []

    for inicio in range(0, len(audio_preenfatizado) - N + 1, M):
        marco = audio_preenfatizado[inicio:inicio + N]
        marcos_listos.append(marco * ventana_hamming)

    return np.asarray(marcos_listos, dtype=np.float64)


def procesar_audios(
    audios: Optional[Dict[str, List[dict]]] = None,
    coeficiente_preenfasis: float = 0.95,
    N: int = 160,
    M: int = 64,
) -> Dict[str, List[AudioProcesado]]:
    """
    Aplica preénfasis y ventaneo a todos los audios.

    Returns:
        {
            etiqueta: [
                {
                    "etiqueta": str,
                    "archivo": str,
                    "ruta": str,
                    "sr": int,
                    "preenfatizado": np.ndarray,
                    "marcos": np.ndarray
                },
                ...
            ],
            ...
        }
    """
    if audios is None:
        audios = muestreo_audios()

    senales_procesadas: Dict[str, List[AudioProcesado]] = {}

    for etiqueta, lista_audios in audios.items():
        senales_procesadas[etiqueta] = []

        for audio_info in lista_audios:
            audio = audio_info["senal"]
            audio_pe = aplicar_preenfasis(audio, coeficiente=coeficiente_preenfasis)
            marcos = aplicar_ventaneo(audio_pe, N=N, M=M)

            senales_procesadas[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": audio_info["archivo"],
                    "ruta": audio_info["ruta"],
                    "sr": audio_info["sr"],
                    "preenfatizado": audio_pe,
                    "marcos": marcos,
                    "num_marcos": len(marcos),
                }
            )

    return senales_procesadas


if __name__ == "__main__":
    senales_procesadas = procesar_audios()
    print("=== Punto 4: Audios procesados ===")
    for etiqueta, lista_audios in senales_procesadas.items():
        print(f"Etiqueta {etiqueta}: {len(lista_audios)} audios")
        if lista_audios:
            print(f"  Marcos del primer audio: {lista_audios[0]['num_marcos']}")
