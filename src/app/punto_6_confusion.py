"""
Punto 6 - Matriz de confusión por AUDIO COMPLETO.

Rediseño:
- Antes se sumaba una predicción por cada vector LPC/marco.
- Ahora se sumará una sola predicción por cada audio.
- Cada audio se clasifica usando la menor distorsión promedio contra cada cuantizador.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.app.punto_4_hamming_filtro import procesar_audios
from src.app.punto_5_cuantizadores import (
    AudioLPC,
    Cuantizador,
    clasificar_audio_por_distorsion,
    entrenar_cuantizadores,
    extraer_vectores_lpc,
)

ResultadoAudio = Dict[str, object]


def extraer_vectores_prueba(
    senales_dict: dict,
    orden: int = 12,
    skip_first: int = 10,
    max_archivos: int = 5,
) -> Dict[str, List[AudioLPC]]:
    """
    Extrae LPC de audios de prueba.

    Por defecto:
        - salta los primeros 10 audios de cada etiqueta,
        - usa los siguientes 5 como prueba.
    """
    return extraer_vectores_lpc(
        senales_dict,
        orden=orden,
        inicio=skip_first,
        max_archivos=max_archivos,
    )


def construir_matriz_confusion_por_audio(
    vectores_prueba: Dict[str, List[AudioLPC]],
    cuantizadores: Dict[str, Cuantizador],
    etiquetas: List[str],
) -> Tuple[np.ndarray, List[ResultadoAudio]]:
    """
    Construye la matriz de confusión evaluando una vez por audio.

    Filas: etiqueta real.
    Columnas: etiqueta predicha.
    """
    matriz_confusion = np.zeros((len(etiquetas), len(etiquetas)), dtype=int)
    etiqueta_a_indice = {etiqueta: idx for idx, etiqueta in enumerate(etiquetas)}
    resultados: List[ResultadoAudio] = []

    for etiqueta_real, lista_audios_lpc in vectores_prueba.items():
        idx_real = etiqueta_a_indice[etiqueta_real]

        for audio_info in lista_audios_lpc:
            etiqueta_predicha, distorsiones = clasificar_audio_por_distorsion(
                audio_info["lpc"],
                cuantizadores,
            )

            idx_predicho = etiqueta_a_indice[etiqueta_predicha]
            matriz_confusion[idx_real, idx_predicho] += 1

            resultados.append(
                {
                    "archivo": audio_info["archivo"],
                    "etiqueta_real": etiqueta_real,
                    "etiqueta_predicha": etiqueta_predicha,
                    "correcto": etiqueta_real == etiqueta_predicha,
                    "num_marcos": audio_info["num_marcos"],
                    "distorsiones": distorsiones,
                    "distorsion_ganadora": distorsiones[etiqueta_predicha],
                }
            )

    return matriz_confusion, resultados


# Alias útil si quieres conservar el nombre anterior, pero ahora clasifica por audio.
def construir_matriz_confusion(
    vectores_prueba: Dict[str, List[AudioLPC]],
    cuantizadores: Dict[str, Cuantizador],
    etiquetas: List[str],
) -> np.ndarray:
    matriz_confusion, _ = construir_matriz_confusion_por_audio(
        vectores_prueba,
        cuantizadores,
        etiquetas,
    )
    return matriz_confusion


def calcular_metricas(matriz_confusion: np.ndarray) -> dict:
    """
    Calcula accuracy, precisión y recall a partir de una matriz de confusión.
    """
    total = int(np.sum(matriz_confusion))
    correctos = int(np.trace(matriz_confusion))
    accuracy = correctos / total if total > 0 else 0

    precision_por_clase = []
    recall_por_clase = []

    for i in range(matriz_confusion.shape[0]):
        tp = matriz_confusion[i, i]

        predichos_clase_i = np.sum(matriz_confusion[:, i])
        precision = tp / predichos_clase_i if predichos_clase_i > 0 else 0
        precision_por_clase.append(float(precision))

        reales_clase_i = np.sum(matriz_confusion[i, :])
        recall = tp / reales_clase_i if reales_clase_i > 0 else 0
        recall_por_clase.append(float(recall))

    return {
        "total_audios_evaluados": total,
        "correctos": correctos,
        "accuracy": accuracy,
        "precision_promedio": float(np.mean(precision_por_clase)) if precision_por_clase else 0,
        "recall_promedio": float(np.mean(recall_por_clase)) if recall_por_clase else 0,
        "precision_por_clase": precision_por_clase,
        "recall_por_clase": recall_por_clase,
    }


def imprimir_matriz_confusion(matriz_confusion: np.ndarray, etiquetas: List[str]) -> None:
    print("\n--- MATRIZ DE CONFUSIÓN POR AUDIO ---")
    print("Filas: Etiqueta real | Columnas: Etiqueta predicha\n")

    print("        ", end="")
    for etiqueta in etiquetas:
        print(f"{str(etiqueta):>8}", end="")
    print()

    for i, etiqueta_real in enumerate(etiquetas):
        print(f"{str(etiqueta_real):>7} ", end="")
        for j in range(len(etiquetas)):
            print(f"{matriz_confusion[i, j]:>8}", end="")
        print()


def imprimir_resultados_por_audio(resultados: List[ResultadoAudio], limite: int = 50) -> None:
    print("\n--- RESULTADOS POR AUDIO ---")
    print("Cada fila es una predicción final de un archivo completo.\n")

    for resultado in resultados[:limite]:
        estado = "✓" if resultado["correcto"] else "✗"
        print(
            f"{estado} archivo={resultado['archivo']} | "
            f"real={resultado['etiqueta_real']} | "
            f"predicha={resultado['etiqueta_predicha']} | "
            f"marcos={resultado['num_marcos']} | "
            f"distorsión={resultado['distorsion_ganadora']:.6f}"
        )

    if len(resultados) > limite:
        print(f"... se omitieron {len(resultados) - limite} resultados más.")


def main():
    print("=== Punto 6: Matriz de Confusión por Audio ===\n")

    orden_lpc = 12
    num_centroides = 256
    num_audios_entrenamiento = 10
    num_audios_prueba = None

    print("Procesando audios...")
    senales_procesadas = procesar_audios()

    print("\nEntrenando cuantizadores con audios de entrenamiento...")
    _, cuantizadores = entrenar_cuantizadores(
        senales_procesadas=senales_procesadas,
        orden=orden_lpc,
        num_centroides=num_centroides,
        num_audios_entrenamiento=num_audios_entrenamiento,
    )

    print("\nExtrayendo LPC de audios de prueba...")
    vectores_prueba = extraer_vectores_prueba(
        senales_procesadas,
        orden=orden_lpc,
        skip_first=0,
        max_archivos=num_audios_prueba,
    )

    etiquetas = sorted(list(cuantizadores.keys()))
    print(f"Etiquetas encontradas: {etiquetas}")

    print("\nClasificando audios completos...")
    matriz_confusion, resultados = construir_matriz_confusion_por_audio(
        vectores_prueba,
        cuantizadores,
        etiquetas,
    )

    imprimir_resultados_por_audio(resultados)
    imprimir_matriz_confusion(matriz_confusion, etiquetas)

    metricas = calcular_metricas(matriz_confusion)

    print("\n--- MÉTRICAS DE DESEMPEÑO POR AUDIO ---")
    print(f"Audios evaluados: {metricas['total_audios_evaluados']}")
    print(f"Audios correctos: {metricas['correctos']}")
    print(f"Accuracy global: {metricas['accuracy']:.4f}")
    print(f"Precisión promedio: {metricas['precision_promedio']:.4f}")
    print(f"Recall promedio: {metricas['recall_promedio']:.4f}")

    print("\n--- MÉTRICAS POR CLASE ---")
    for idx, etiqueta in enumerate(etiquetas):
        print(
            f"{etiqueta}: "
            f"Precision={metricas['precision_por_clase'][idx]:.4f}, "
            f"Recall={metricas['recall_por_clase'][idx]:.4f}"
        )

    print("\n✓ Punto 6 completado")
    return matriz_confusion, metricas, resultados


if __name__ == "__main__":
    matriz_confusion, metricas, resultados = main()
