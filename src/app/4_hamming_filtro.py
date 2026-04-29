
import numpy as np
from muestreo import muestreo_audios
from typing import List
from scipy import signal

def aplicar_preenfasis(audio, coeficiente=0.95): 
    audio_preenfatizado = np.zeros_like(audio)
    audio_preenfatizado[0] = audio[0]
    for n in range(1, len(audio)):
        audio_preenfatizado[n] = audio[n] - coeficiente * audio[n - 1]
    return audio_preenfatizado


def aplicar_ventaneo(audio_preenfatizado : np.ndarray[float], N = 160, M = 64) -> np.ndarray[List[float]]:
    marcos_listos = []
    n = np.arange(N)
    ventana_hamming = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))
        
    # Se corre cada M muestras
    for inicio in range(0, len(audio_preenfatizado) - N + 1, M):
        marco = audio_preenfatizado[inicio:inicio + N]
        marco_suavizado = marco * ventana_hamming
        marcos_listos.append(marco_suavizado)
            
    return np.array(marcos_listos)



audios = muestreo_audios()
senales_procesadas = {}

for etiqueta, lista_audios in audios.items():
    senales_procesadas[etiqueta] = []
    for audio in lista_audios:
        audio_pe = aplicar_preenfasis(audio)
        marcos = aplicar_ventaneo(audio_pe)
        senales_procesadas[etiqueta].append({
            "preenfatizado": audio_pe,
            "marcos": marcos
        })

