import os
import re
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np


# python -m src.app.punto_6_confusion
ruta_base_audio = "src/resources/audio"

AudioCrudo = Dict[str, object]


def obtener_clave_orden_natural(nombre: str) -> Tuple[object, ...]:
    """
    Ordena cadenas con números de forma natural.

    Ejemplo:
        cinco_1.wav, cinco_2.wav, ..., cinco_10.wav

    Sin esto, Python ordena lexicográficamente:
        cinco_10.wav, cinco_11.wav, ..., cinco_1.wav
    """
    partes = re.split(r"(\d+)", nombre)
    return tuple(int(parte) if parte.isdigit() else parte.lower() for parte in partes)


def cargar_audios(ruta_base: Optional[str] = None, frecuencia_muestreo: Optional[int] = None) -> Dict[str, List[AudioCrudo]]:
    """
    Carga audios WAV agrupados por etiqueta.

    Estructura esperada:
        src/resources/audio/<etiqueta>/<archivo>.wav

    Argumentos:
        ruta_base: Ruta base donde están las carpetas de etiquetas.
        frecuencia_muestreo: Frecuencia de muestreo deseada. Si es None, conserva la original.

    Retorna:
        {
            etiqueta: [
                {
                    "etiqueta": str,
                    "archivo": str,
                    "ruta": str,
                    "senal": np.ndarray,
                    "frecuencia_muestreo": int,
                    "duracion_segundos": float,
                },
                ...
            ],
            ...
        }
    """
    if ruta_base is None:
        ruta_base = ruta_base_audio

    if not os.path.isdir(ruta_base):
        raise FileNotFoundError(
            f"No existe la carpeta de audios: {ruta_base}. "
            "Ejecuta desde la raíz del proyecto o ajusta ruta_base_audio."
        )

    audios: Dict[str, List[AudioCrudo]] = {}

    for etiqueta in sorted(os.listdir(ruta_base), key=obtener_clave_orden_natural):
        ruta_etiqueta = os.path.join(ruta_base, etiqueta)

        if not os.path.isdir(ruta_etiqueta):
            continue

        audios[etiqueta] = []

        for archivo in sorted(os.listdir(ruta_etiqueta), key=obtener_clave_orden_natural):
            if not archivo.lower().endswith(".wav"):
                continue

            ruta_completa = os.path.join(ruta_etiqueta, archivo)
            senal, frecuencia_real = librosa.load(ruta_completa, sr=frecuencia_muestreo, mono=True)
            senal = np.asarray(senal, dtype=np.float64)

            audios[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": archivo,
                    "ruta": ruta_completa,
                    "senal": senal,
                    "frecuencia_muestreo": frecuencia_real,
                    "duracion_segundos": len(senal) / frecuencia_real if frecuencia_real else 0.0,
                }
            )

    return audios


if __name__ == "__main__":
    audios = cargar_audios()
    print("=== Punto 3: Audios cargados ===")
    for etiqueta, lista_audios in audios.items():
        archivos = [audio["archivo"] for audio in lista_audios]
        print(f"Etiqueta {etiqueta}: {len(lista_audios)} audios")
        print(f"  Orden: {archivos}")
