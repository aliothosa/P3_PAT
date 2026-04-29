"""
Punto 3 - Muestreo / carga de audios.

Cambios importantes:
- Conserva metadatos por audio: etiqueta, archivo, ruta, señal y frecuencia de muestreo.
- Usa orden natural de archivos: uno_1.wav, uno_2.wav, ..., uno_10.wav.
  Esto evita que sorted() ponga uno_10.wav antes de uno_1.wav.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np


# Ruta esperada si ejecutas desde la raíz del proyecto con:
# python -m src.app.punto_6_confusion
audio_base_path = "src/resources/audio"

AudioCrudo = Dict[str, object]


def clave_orden_natural(nombre: str) -> Tuple[object, ...]:
    """
    Ordena cadenas con números de forma natural.

    Ejemplo:
        cinco_1.wav, cinco_2.wav, ..., cinco_10.wav

    Sin esto, Python ordena lexicográficamente:
        cinco_10.wav, cinco_11.wav, ..., cinco_1.wav
    """
    partes = re.split(r"(\d+)", nombre)
    return tuple(int(parte) if parte.isdigit() else parte.lower() for parte in partes)


def muestreo_audios(
    base_path: Optional[str] = None,
    sr: Optional[int] = None,
) -> Dict[str, List[AudioCrudo]]:
    """
    Carga audios WAV agrupados por etiqueta.

    Estructura esperada:
        src/resources/audio/<etiqueta>/<archivo>.wav

    Args:
        base_path: Ruta base donde están las carpetas de etiquetas.
        sr: Frecuencia de muestreo deseada. Si es None, conserva la original.

    Returns:
        {
            etiqueta: [
                {
                    "etiqueta": str,
                    "archivo": str,
                    "ruta": str,
                    "senal": np.ndarray,
                    "sr": int,
                    "duracion_segundos": float,
                },
                ...
            ],
            ...
        }
    """
    if base_path is None:
        base_path = audio_base_path

    if not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"No existe la carpeta de audios: {base_path}. "
            "Ejecuta desde la raíz del proyecto o ajusta audio_base_path."
        )

    audios: Dict[str, List[AudioCrudo]] = {}

    for etiqueta in sorted(os.listdir(base_path), key=clave_orden_natural):
        etiqueta_path = os.path.join(base_path, etiqueta)

        if not os.path.isdir(etiqueta_path):
            continue

        audios[etiqueta] = []

        for archivo in sorted(os.listdir(etiqueta_path), key=clave_orden_natural):
            if not archivo.lower().endswith(".wav"):
                continue

            ruta_completa = os.path.join(etiqueta_path, archivo)
            senal, sr_real = librosa.load(ruta_completa, sr=sr, mono=True)
            senal = np.asarray(senal, dtype=np.float64)

            audios[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": archivo,
                    "ruta": ruta_completa,
                    "senal": senal,
                    "sr": sr_real,
                    "duracion_segundos": len(senal) / sr_real if sr_real else 0.0,
                }
            )

    return audios


if __name__ == "__main__":
    audios = muestreo_audios()
    print("=== Punto 3: Audios cargados ===")
    for etiqueta, lista_audios in audios.items():
        archivos = [audio["archivo"] for audio in lista_audios]
        print(f"Etiqueta {etiqueta}: {len(lista_audios)} audios")
        print(f"  Orden: {archivos}")
