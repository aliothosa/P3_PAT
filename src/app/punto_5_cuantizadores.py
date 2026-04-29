"""
Punto 5 - LPC y cuantizadores vectoriales LBG.

Rediseño:
- El cuantizador se entrena con vectores LPC de los audios de entrenamiento.
- Se agrega una función para clasificar un AUDIO COMPLETO, no solo un marco.
- La salida por audio se decide por la menor distorsión promedio contra cada cuantizador.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np

from src.app.punto_4_hamming_filtro import procesar_audios

# Importar filtroWiener desde utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.filtroWiener import filtroWiener  # noqa: E402

AudioLPC = Dict[str, object]
Cuantizador = Dict[str, object]


def calcular_coeficientes_lpc(marco: np.ndarray, orden: int = 12) -> np.ndarray:
    """
    Calcula coeficientes LPC de un marco usando filtroWiener.

    Args:
        marco: Marco de audio de N muestras.
        orden: Número de coeficientes LPC.

    Returns:
        Vector LPC con dimensión `orden`.
    """
    marco = np.asarray(marco, dtype=np.float64)

    if len(marco) <= orden:
        return np.zeros(orden, dtype=np.float64)

    coeficientes, _, _ = filtroWiener(marco.tolist(), marco.tolist(), orden)
    return np.asarray(coeficientes, dtype=np.float64)


def distancia_itakura_saito(lpc1: np.ndarray, lpc2: np.ndarray) -> float:
    """
    Distancia usada por el proyecto.

    Nota: esta función conserva el nombre original, pero implementa una distancia
    cuadrática normalizada entre vectores LPC.
    """
    lpc1 = np.asarray(lpc1, dtype=np.float64)
    lpc2 = np.asarray(lpc2, dtype=np.float64)

    lpc2_safe = np.where(np.abs(lpc2) < 1e-10, 1e-10, lpc2)
    distancia = np.sum(((lpc1 - lpc2) ** 2) / ((lpc2_safe ** 2) + 1e-10))

    return float(np.sqrt(distancia))


def distancias_a_centroides(vector_lpc: np.ndarray, centroides: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de un vector LPC contra todos los centroides.
    """
    vector_lpc = np.asarray(vector_lpc, dtype=np.float64)
    centroides = np.asarray(centroides, dtype=np.float64)

    centroides_safe = np.where(np.abs(centroides) < 1e-10, 1e-10, centroides)
    distancias = np.sqrt(
        np.sum(((vector_lpc - centroides) ** 2) / ((centroides_safe ** 2) + 1e-10), axis=1)
    )

    return distancias


def extraer_lpc_de_audio(audio_procesado: dict, orden: int = 12) -> np.ndarray:
    """
    Convierte todos los marcos de un audio procesado en vectores LPC.

    Returns:
        Matriz con forma (num_marcos, orden).
    """
    marcos = audio_procesado["marcos"]
    coeficientes_por_marco = [calcular_coeficientes_lpc(marco, orden) for marco in marcos]

    return np.asarray(coeficientes_por_marco, dtype=np.float64)


def extraer_vectores_lpc(
    senales_dict: Dict[str, List[dict]],
    orden: int = 12,
    inicio: int = 0,
    max_archivos: Optional[int] = 10,
) -> Dict[str, List[AudioLPC]]:
    """
    Extrae LPC por audio conservando metadatos.

    Args:
        senales_dict: Señales procesadas por etiqueta.
        orden: Orden LPC.
        inicio: Índice inicial de archivos por etiqueta.
        max_archivos: Cantidad máxima de archivos a tomar. Si es None, toma todos.

    Returns:
        {
            etiqueta: [
                {
                    "etiqueta": str,
                    "archivo": str,
                    "ruta": str,
                    "sr": int,
                    "lpc": np.ndarray,        # (num_marcos, orden)
                    "num_marcos": int
                },
                ...
            ],
            ...
        }
    """
    vectores_lpc: Dict[str, List[AudioLPC]] = {}

    for etiqueta, lista_audios in senales_dict.items():
        fin = None if max_archivos is None else inicio + max_archivos
        audios_seleccionados = lista_audios[inicio:fin]
        vectores_lpc[etiqueta] = []

        for audio_procesado in audios_seleccionados:
            lpc_audio = extraer_lpc_de_audio(audio_procesado, orden=orden)

            vectores_lpc[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": audio_procesado["archivo"],
                    "ruta": audio_procesado["ruta"],
                    "sr": audio_procesado["sr"],
                    "lpc": lpc_audio,
                    "num_marcos": len(lpc_audio),
                }
            )

    return vectores_lpc


def crear_cuantizador_lbg(
    vectores: np.ndarray,
    num_centroides: int,
    max_iteraciones: int = 100,
    epsilon: float = 0.01,
    perturbacion: float = 0.01,
) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo LBG para crear un cuantizador vectorial.

    El número de centroides debe ser potencia de 2 porque el algoritmo va duplicando:
        1 -> 2 -> 4 -> ... -> num_centroides
    """
    vectores = np.asarray(vectores, dtype=np.float64)

    if len(vectores) == 0:
        raise ValueError("No hay vectores para entrenar el cuantizador.")

    if num_centroides < 1 or int(np.log2(num_centroides)) != np.log2(num_centroides):
        raise ValueError("num_centroides debe ser potencia de 2: 1, 2, 4, 8, ..., 256.")

    centroides = np.asarray([np.mean(vectores, axis=0)], dtype=np.float64)
    distorsiones: List[float] = []

    for _ in range(int(np.log2(num_centroides))):
        centroides = np.asarray(
            [nuevo for centroide in centroides
             for nuevo in (centroide * (1 + perturbacion), centroide * (1 - perturbacion))],
            dtype=np.float64,
        )

        for _ in range(max_iteraciones):
            matriz_distancias = np.asarray(
                [distancias_a_centroides(vector, centroides) for vector in vectores],
                dtype=np.float64,
            )

            asignaciones = np.argmin(matriz_distancias, axis=1)
            distorsion = float(np.mean(np.min(matriz_distancias, axis=1)))
            distorsiones.append(distorsion)

            nuevos_centroides = []
            for k in range(len(centroides)):
                vectores_asignados = vectores[asignaciones == k]

                if len(vectores_asignados) > 0:
                    nuevos_centroides.append(np.mean(vectores_asignados, axis=0))
                else:
                    nuevos_centroides.append(centroides[k])

            nuevos_centroides = np.asarray(nuevos_centroides, dtype=np.float64)

            cambio = np.mean(
                [distancia_itakura_saito(nuevos_centroides[i], centroides[i])
                 for i in range(len(centroides))]
            )

            centroides = nuevos_centroides

            if cambio < epsilon:
                break

    return centroides, distorsiones


def entrenar_cuantizadores(
    senales_procesadas: Optional[Dict[str, List[dict]]] = None,
    orden: int = 12,
    num_centroides: int = 256,
    num_audios_entrenamiento: int = 10,
    max_iteraciones: int = 100,
    epsilon: float = 0.01,
) -> Tuple[Dict[str, List[AudioLPC]], Dict[str, Cuantizador]]:
    """
    Entrena un cuantizador por etiqueta.

    Returns:
        vectores_entrenamiento, cuantizadores
    """
    if senales_procesadas is None:
        senales_procesadas = procesar_audios()

    vectores_entrenamiento = extraer_vectores_lpc(
        senales_procesadas,
        orden=orden,
        inicio=0,
        max_archivos=num_audios_entrenamiento,
    )

    cuantizadores: Dict[str, Cuantizador] = {}

    for etiqueta, audios_lpc in vectores_entrenamiento.items():
        if not audios_lpc:
            continue

        todos_vectores = np.vstack([audio_info["lpc"] for audio_info in audios_lpc])

        centroides, distorsion = crear_cuantizador_lbg(
            todos_vectores,
            num_centroides=num_centroides,
            max_iteraciones=max_iteraciones,
            epsilon=epsilon,
        )

        cuantizadores[etiqueta] = {
            "etiqueta": etiqueta,
            "centroides": centroides,
            "distorsion_entrenamiento": distorsion,
            "num_audios_entrenamiento": len(audios_lpc),
            "num_vectores_entrenamiento": len(todos_vectores),
            "orden_lpc": orden,
        }

    return vectores_entrenamiento, cuantizadores


def calcular_distorsion_audio(vectores_lpc_audio: np.ndarray, centroide_codigo: np.ndarray) -> float:
    """
    Calcula qué tan bien un cuantizador representa un audio completo.

    Para cada marco/vector LPC del audio:
        1. Busca el centroide más cercano dentro del cuantizador.
        2. Guarda esa distancia mínima.

    La distorsión del audio contra ese cuantizador es el promedio de esas distancias.
    """
    vectores_lpc_audio = np.asarray(vectores_lpc_audio, dtype=np.float64)

    if len(vectores_lpc_audio) == 0:
        return float("inf")

    distancias_minimas = [
        float(np.min(distancias_a_centroides(vector_lpc, centroide_codigo)))
        for vector_lpc in vectores_lpc_audio
    ]

    return float(np.mean(distancias_minimas))


def clasificar_audio_por_distorsion(
    vectores_lpc_audio: np.ndarray,
    cuantizadores: Dict[str, Cuantizador],
) -> Tuple[str, Dict[str, float]]:
    """
    Clasifica un AUDIO COMPLETO.

    La predicción es la etiqueta cuyo cuantizador tenga menor distorsión promedio
    sobre todos los vectores LPC del audio.

    Returns:
        etiqueta_predicha, distorsiones_por_etiqueta
    """
    distorsiones_por_etiqueta: Dict[str, float] = {}

    for etiqueta, cuantizador in cuantizadores.items():
        distorsiones_por_etiqueta[etiqueta] = calcular_distorsion_audio(
            vectores_lpc_audio,
            cuantizador["centroides"],
        )

    etiqueta_predicha = min(distorsiones_por_etiqueta, key=distorsiones_por_etiqueta.get)

    return etiqueta_predicha, distorsiones_por_etiqueta


def main():
    print("=== Punto 5: Cuantizadores Vectoriales con LPC ===\n")

    senales_procesadas = procesar_audios()
    vectores_entrenamiento, cuantizadores = entrenar_cuantizadores(
        senales_procesadas=senales_procesadas,
        orden=12,
        num_centroides=256,
        num_audios_entrenamiento=10,
    )

    for etiqueta, cuantizador in cuantizadores.items():
        print(f"\nEtiqueta: {etiqueta}")
        print(f"  Audios entrenamiento: {cuantizador['num_audios_entrenamiento']}")
        print(f"  Vectores LPC entrenamiento: {cuantizador['num_vectores_entrenamiento']}")
        print(f"  Centroides generados: {len(cuantizador['centroides'])}")
        print(f"  Distorsión final entrenamiento: {cuantizador['distorsion_entrenamiento'][-1]:.6f}")

    print("\n✓ Punto 5 completado")
    return vectores_entrenamiento, cuantizadores


if __name__ == "__main__":
    vectores_lpc, cuantizadores = main()
