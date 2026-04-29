import os
import librosa
import numpy as np

# Ruta base de los audios (relativa al directorio del script)
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_base_path = os.path.join(script_dir, "..", "resources", "audio")

def muestreo_audios() -> dict:
    audios = {}

    for etiqueta in os.listdir(audio_base_path):
        etiqueta_path = os.path.join(audio_base_path, etiqueta)
        if os.path.isdir(etiqueta_path):
            audios[etiqueta] = []
            # Recorrer archivos .wav en la subcarpeta
            for archivo in os.listdir(etiqueta_path):
                if archivo.endswith(".wav"):
                    ruta_completa = os.path.join(etiqueta_path, archivo)
                    # Cargar audio a frecuencia original 
                    y, sr = librosa.load(ruta_completa, sr=None)  # sr=None mantiene 16 kHz
                    audios[etiqueta].append(y)

        
    return audios
