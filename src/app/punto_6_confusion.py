"""
Punto 6 - Matriz de confusión por AUDIO COMPLETO.

Cambios importantes:
- La matriz suma una sola predicción por audio.
- Cada audio se clasifica usando la distancia Itakura-Saito promedio recortada
  de sus marcos útiles.
- Se muestran diagnósticos del recorte por potencia y el top 3 de distorsiones.
"""

from typing import Dict, List, Optional, Tuple

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

ORDEN_ETIQUETAS = [
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
]


def ordenar_etiquetas(etiquetas: List[str]) -> List[str]:
    conocidas = [etiqueta for etiqueta in ORDEN_ETIQUETAS if etiqueta in etiquetas]
    restantes = sorted([etiqueta for etiqueta in etiquetas if etiqueta not in ORDEN_ETIQUETAS])
    return conocidas + restantes


def extraer_vectores_prueba(
    senales_dict: dict,
    orden: int = 12,
    skip_first: int = 10,
    max_archivos: Optional[int] = 5,
) -> Dict[str, List[AudioLPC]]:
    """
    Extrae LPC/autocorrelaciones de audios de prueba.

    Para probar todos los audios:
        skip_first=0
        max_archivos=None
    """
    return extraer_vectores_lpc(
        senales_dict,
        orden=orden,
        inicio=skip_first,
        max_archivos=max_archivos,
    )


def top_distorsiones(distorsiones: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(distorsiones.items(), key=lambda item: item[1])[:k]


def construir_matriz_confusion_por_audio(
    vectores_prueba: Dict[str, List[AudioLPC]],
    cuantizadores: Dict[str, Cuantizador],
    etiquetas: List[str],
    porcentaje_recorte_superior: float = 0.10,
) -> Tuple[np.ndarray, List[ResultadoAudio]]:
    """
    Construye matriz de confusión evaluando una vez por audio.

    Filas: etiqueta real.
    Columnas: etiqueta predicha.
    """
    matriz_confusion = np.zeros((len(etiquetas), len(etiquetas)), dtype=int)
    etiqueta_a_indice = {etiqueta: idx for idx, etiqueta in enumerate(etiquetas)}
    resultados: List[ResultadoAudio] = []

    for etiqueta_real, lista_audios_lpc in vectores_prueba.items():
        if etiqueta_real not in etiqueta_a_indice:
            continue

        idx_real = etiqueta_a_indice[etiqueta_real]

        for audio_info in lista_audios_lpc:
            if len(audio_info["autocorr"]) == 0:
                resultados.append(
                    {
                        "archivo": audio_info["archivo"],
                        "etiqueta_real": etiqueta_real,
                        "etiqueta_predicha": None,
                        "correcto": False,
                        "num_marcos": 0,
                        "distorsiones": {},
                        "distorsion_ganadora": float("inf"),
                        "top3": [],
                        "recorte_inicio": audio_info.get("recorte_inicio"),
                        "recorte_fin": audio_info.get("recorte_fin"),
                    }
                )
                continue

            etiqueta_predicha, distorsiones = clasificar_audio_por_distorsion(
                audio_info["autocorr"],
                cuantizadores,
                porcentaje_recorte_superior=porcentaje_recorte_superior,
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
                    "num_marcos_antes_filtro": audio_info.get("num_marcos_antes_filtro"),
                    "marcos_descartados": audio_info.get("marcos_descartados", 0),
                    "recorte_inicio": audio_info.get("recorte_inicio"),
                    "recorte_fin": audio_info.get("recorte_fin"),
                    "distorsiones": distorsiones,
                    "distorsion_ganadora": distorsiones[etiqueta_predicha],
                    "top3": top_distorsiones(distorsiones, k=3),
                }
            )

    return matriz_confusion, resultados


# Alias por compatibilidad con el nombre anterior.
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
    total = int(np.sum(matriz_confusion))
    correctos = int(np.trace(matriz_confusion))
    accuracy = correctos / total if total > 0 else 0.0

    precision_por_clase = []
    recall_por_clase = []

    for i in range(matriz_confusion.shape[0]):
        tp = matriz_confusion[i, i]

        predichos_clase_i = np.sum(matriz_confusion[:, i])
        precision = tp / predichos_clase_i if predichos_clase_i > 0 else 0.0
        precision_por_clase.append(float(precision))

        reales_clase_i = np.sum(matriz_confusion[i, :])
        recall = tp / reales_clase_i if reales_clase_i > 0 else 0.0
        recall_por_clase.append(float(recall))

    return {
        "total_audios_evaluados": total,
        "correctos": correctos,
        "accuracy": accuracy,
        "precision_promedio": float(np.mean(precision_por_clase)) if precision_por_clase else 0.0,
        "recall_promedio": float(np.mean(recall_por_clase)) if recall_por_clase else 0.0,
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


def imprimir_resultados_por_audio(resultados: List[ResultadoAudio], limite: int = 150) -> None:
    print("\n--- RESULTADOS POR AUDIO ---")
    print("Cada fila es una predicción final de un archivo completo.\n")

    for resultado in resultados[:limite]:
        estado = "✓" if resultado["correcto"] else "✗"
        top3 = resultado.get("top3", [])
        top3_txt = ", ".join([f"{etq}:{dist:.4f}" for etq, dist in top3])

        print(
            f"{estado} archivo={resultado['archivo']} | "
            f"real={resultado['etiqueta_real']} | "
            f"predicha={resultado['etiqueta_predicha']} | "
            f"marcos={resultado['num_marcos']} | "
            f"recorte={resultado.get('recorte_inicio')}:{resultado.get('recorte_fin')} | "
            f"top3=[{top3_txt}]"
        )

    if len(resultados) > limite:
        print(f"... se omitieron {len(resultados) - limite} resultados más.")


def main():
    print("=== Punto 6: Matriz de Confusión por Audio ===\n")

    # Configuración principal.
    orden_lpc = 12
    num_centroides = 256
    num_audios_entrenamiento = 10

    # Para evaluar todos los audios, incluyendo los de entrenamiento:
    probar_todos_los_audios = True

    if probar_todos_los_audios:
        skip_first_prueba = 0
        num_audios_prueba = None
    else:
        skip_first_prueba = num_audios_entrenamiento
        num_audios_prueba = 5

    print("Procesando audios con recorte por potencia...")
    senales_procesadas = procesar_audios(
        N=160,
        M=64,
        usar_recorte_por_potencia=True,
        umbral_recorte=0.12,
        margen_marcos=8,
        filtrar_marcos_bajos=True,
        umbral_marcos=0.04,
    )

    print("\nEntrenando cuantizadores con audios de entrenamiento...")
    _, cuantizadores = entrenar_cuantizadores(
        senales_procesadas=senales_procesadas,
        orden=orden_lpc,
        num_centroides=num_centroides,
        num_audios_entrenamiento=num_audios_entrenamiento,
        max_iteraciones=80,
        epsilon=1e-4,
    )

    print("\nResumen de cuantizadores:")
    for etiqueta, cuantizador in cuantizadores.items():
        print(
            f"  {etiqueta}: "
            f"audios={cuantizador['num_audios_entrenamiento']} | "
            f"vectores={cuantizador['num_vectores_entrenamiento']} | "
            f"centroides={cuantizador['num_centroides_reales']}"
        )

    print("\nExtrayendo LPC/autocorrelaciones de audios de prueba...")
    vectores_prueba = extraer_vectores_prueba(
        senales_procesadas,
        orden=orden_lpc,
        skip_first=skip_first_prueba,
        max_archivos=num_audios_prueba,
    )

    etiquetas = ordenar_etiquetas(list(cuantizadores.keys()))
    print(f"Etiquetas encontradas: {etiquetas}")

    print("\nClasificando audios completos...")
    matriz_confusion, resultados = construir_matriz_confusion_por_audio(
        vectores_prueba,
        cuantizadores,
        etiquetas,
        porcentaje_recorte_superior=0.10,
    )

    imprimir_resultados_por_audio(resultados)
    imprimir_matriz_confusion(matriz_confusion, etiquetas)

    metricas = calcular_metricas(matriz_confusion)

    print("\n--- MÉTRICAS DE DESEMPEÑO POR AUDIO ---")
    print(f"Audios evaluados: {metricas['total_audios_evaluados']}")
    print(f"Audios correctos: {metricas['correctos']}")
    print(f"Precisión global: {metricas['accuracy']:.4f}")
    print(f"Precisión promedio: {metricas['precision_promedio']:.4f}")

    print("\n--- MÉTRICAS POR CLASE ---")
    for idx, etiqueta in enumerate(etiquetas):
        print(
            f"{etiqueta}: "
            f"Precision={metricas['precision_por_clase'][idx]:.4f}, "
        )

    print("\n✓ Punto 6 completado")
    return matriz_confusion, metricas, resultados


if __name__ == "__main__":
    matriz_confusion, metricas, resultados = main()
