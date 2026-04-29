"""
Punto 3 - Muestreo / carga de audios.

Rediseño:
- Conserva metadatos por audio: etiqueta, archivo, ruta, señal y frecuencia de muestreo.
- Ordena etiquetas y archivos para que el split entrenamiento/prueba sea reproducible.
"""

import os
from typing import Dict, List, Optional

import librosa
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
audio_base_path = "src/resources/audio"  # Ruta relativa a este script, se puede ajustar si es necesario

AudioCrudo = Dict[str, object]


def muestreo_audios(
    base_path: Optional[str] = None,
    sr: Optional[int] = None,
) -> Dict[str, List[AudioCrudo]]:
    """
    Carga audios WAV agrupados por etiqueta.

    Estructura esperada:
        resources/audio/<etiqueta>/<archivo>.wav

    Args:
        base_path: Ruta base donde están las carpetas de etiquetas.
        sr: Frecuencia de muestreo deseada. Si es None, conserva la original.

    Returns:
        Diccionario:
        {
            etiqueta: [
                {
                    "etiqueta": str,
                    "archivo": str,
                    "ruta": str,
                    "senal": np.ndarray,
                    "sr": int
                },
                ...
            ],
            ...
        }
    """
    if base_path is None:
        base_path = audio_base_path

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"No existe la carpeta de audios: {base_path}")

    audios: Dict[str, List[AudioCrudo]] = {}

    for etiqueta in sorted(os.listdir(base_path)):
        etiqueta_path = os.path.join(base_path, etiqueta)

        if not os.path.isdir(etiqueta_path):
            continue

        audios[etiqueta] = []

        for archivo in sorted(os.listdir(etiqueta_path)):
            if not archivo.lower().endswith(".wav"):
                continue

            ruta_completa = os.path.join(etiqueta_path, archivo)
            senal, sr_real = librosa.load(ruta_completa, sr=sr)

            audios[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": archivo,
                    "ruta": ruta_completa,
                    "senal": np.asarray(senal, dtype=np.float64),
                    "sr": sr_real,
                }
            )

    return audios


if __name__ == "__main__":
    audios = muestreo_audios()
    print("=== Punto 3: Audios cargados ===")
    for etiqueta, lista_audios in audios.items():
        print(f"Etiqueta {etiqueta}: {len(lista_audios)} audios")
