from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


def extraer_vectores_prueba(senales_por_etiqueta: dict, orden: int = 12, omitir_primeros: int = 10, maximo_archivos: Optional[int] = 5) -> Dict[str, List[AudioLPC]]:
    """
    Extrae LPC/autocorrelaciones de audios de prueba.

    Para probar todos los audios:
        omitir_primeros=0
        maximo_archivos=None
    """
    return extraer_vectores_lpc(
        senales_por_etiqueta,
        orden=orden,
        inicio=omitir_primeros,
        maximo_archivos=maximo_archivos,
    )


def obtener_mejores_distorsiones(distorsiones: Dict[str, float], cantidad: int = 3) -> List[Tuple[str, float]]:
    return sorted(distorsiones.items(), key=lambda elemento: elemento[1])[:cantidad]


def construir_matriz_confusion_por_audio(vectores_prueba: Dict[str, List[AudioLPC]], cuantizadores: Dict[str, Cuantizador], etiquetas: List[str], porcentaje_recorte_superior: float = 0.10) -> Tuple[np.ndarray, List[ResultadoAudio]]:
    """
    Construye matriz de confusión evaluando una vez por audio.

    Filas: etiqueta real.
    Columnas: etiqueta predicha.
    """
    matriz_confusion = np.zeros((len(etiquetas), len(etiquetas)), dtype=int)
    etiqueta_a_indice = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
    resultados: List[ResultadoAudio] = []

    for etiqueta_real, lista_audios_lpc in vectores_prueba.items():
        if etiqueta_real not in etiqueta_a_indice:
            continue

        indice_real = etiqueta_a_indice[etiqueta_real]

        for informacion_audio in lista_audios_lpc:
            if len(informacion_audio["autocorrelacion"]) == 0:
                resultados.append(
                    {
                        "archivo": informacion_audio["archivo"],
                        "etiqueta_real": etiqueta_real,
                        "etiqueta_predicha": None,
                        "correcto": False,
                        "numero_marcos": 0,
                        "distorsiones": {},
                        "distorsion_ganadora": float("inf"),
                        "mejores_distorsiones": [],
                        "recorte_inicio": informacion_audio.get("recorte_inicio"),
                        "recorte_fin": informacion_audio.get("recorte_fin"),
                    }
                )
                continue

            etiqueta_predicha, distorsiones = clasificar_audio_por_distorsion(
                informacion_audio["autocorrelacion"],
                cuantizadores,
                porcentaje_recorte_superior=porcentaje_recorte_superior,
            )

            indice_predicho = etiqueta_a_indice[etiqueta_predicha]
            matriz_confusion[indice_real, indice_predicho] += 1

            resultados.append(
                {
                    "archivo": informacion_audio["archivo"],
                    "etiqueta_real": etiqueta_real,
                    "etiqueta_predicha": etiqueta_predicha,
                    "correcto": etiqueta_real == etiqueta_predicha,
                    "numero_marcos": informacion_audio["numero_marcos"],
                    "numero_marcos_antes_filtro": informacion_audio.get("numero_marcos_antes_filtro"),
                    "marcos_descartados": informacion_audio.get("marcos_descartados", 0),
                    "recorte_inicio": informacion_audio.get("recorte_inicio"),
                    "recorte_fin": informacion_audio.get("recorte_fin"),
                    "distorsiones": distorsiones,
                    "distorsion_ganadora": distorsiones[etiqueta_predicha],
                    "mejores_distorsiones": obtener_mejores_distorsiones(distorsiones, cantidad=3),
                }
            )

    return matriz_confusion, resultados


# Alias por compatibilidad con el nombre anterior.
def construir_matriz_confusion(vectores_prueba: Dict[str, List[AudioLPC]], cuantizadores: Dict[str, Cuantizador], etiquetas: List[str]) -> np.ndarray:
    matriz_confusion, _ = construir_matriz_confusion_por_audio(vectores_prueba, cuantizadores, etiquetas)
    return matriz_confusion


def calcular_metricas(matriz_confusion: np.ndarray) -> dict:
    total = int(np.sum(matriz_confusion))
    correctos = int(np.trace(matriz_confusion))
    exactitud = correctos / total if total > 0 else 0.0

    precision_por_clase = []
    sensibilidad_por_clase = []

    for indice in range(matriz_confusion.shape[0]):
        verdaderos_positivos = matriz_confusion[indice, indice]

        predichos_clase = np.sum(matriz_confusion[:, indice])
        precision = verdaderos_positivos / predichos_clase if predichos_clase > 0 else 0.0
        precision_por_clase.append(float(precision))

        reales_clase = np.sum(matriz_confusion[indice, :])
        sensibilidad = verdaderos_positivos / reales_clase if reales_clase > 0 else 0.0
        sensibilidad_por_clase.append(float(sensibilidad))

    return {
        "total_audios_evaluados": total,
        "correctos": correctos,
        "exactitud": exactitud,
        "precision_promedio": float(np.mean(precision_por_clase)) if precision_por_clase else 0.0,
        "sensibilidad_promedio": float(np.mean(sensibilidad_por_clase)) if sensibilidad_por_clase else 0.0,
        "precision_por_clase": precision_por_clase,
        "sensibilidad_por_clase": sensibilidad_por_clase,
    }



def graficar_matriz_confusion(matriz_confusion: np.ndarray, etiquetas: List[str], normalizar: bool = False, titulo: str = "Matriz de confusión por audio", ruta_salida: Optional[str] = None) -> None:
    """
    Muestra la matriz de confusión como mapa de calor con matplotlib.

    Args:
        matriz_confusion: Matriz donde las filas son etiquetas reales y las columnas predichas.
        etiquetas: Nombres de las clases en el mismo orden que la matriz.
        normalizar: Si es True, muestra proporciones por fila en lugar de conteos absolutos.
        titulo: Título de la gráfica.
        ruta_salida: Ruta opcional para guardar la imagen, por ejemplo "matriz_confusion.png".
    """
    matriz = np.asarray(matriz_confusion, dtype=float)

    if normalizar:
        suma_filas = matriz.sum(axis=1, keepdims=True)
        matriz_mostrar = np.divide(matriz, suma_filas, out=np.zeros_like(matriz), where=suma_filas != 0)
        formato_texto = ".2f"
        etiqueta_barra = "Proporción"
    else:
        matriz_mostrar = matriz
        formato_texto = ".0f"
        etiqueta_barra = "Cantidad"

    figura, eje = plt.subplots(figsize=(10, 8))
    imagen = eje.imshow(matriz_mostrar, cmap="Blues")
    barra = figura.colorbar(imagen, ax=eje)
    barra.set_label(etiqueta_barra)

    eje.set_title(titulo)
    eje.set_xlabel("Etiqueta predicha")
    eje.set_ylabel("Etiqueta real")
    eje.set_xticks(np.arange(len(etiquetas)))
    eje.set_yticks(np.arange(len(etiquetas)))
    eje.set_xticklabels(etiquetas, rotation=45, ha="right")
    eje.set_yticklabels(etiquetas)

    limite_color = matriz_mostrar.max() / 2 if matriz_mostrar.size and matriz_mostrar.max() > 0 else 0

    for fila in range(matriz_mostrar.shape[0]):
        for columna in range(matriz_mostrar.shape[1]):
            valor = matriz_mostrar[fila, columna]
            color_texto = "white" if valor > limite_color else "black"
            eje.text(columna, fila, format(valor, formato_texto), ha="center", va="center", color=color_texto)

    figura.tight_layout()

    if ruta_salida:
        figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
        print(f"Gráfica guardada en: {ruta_salida}")
        
    plt.savefig("src/output/" + ruta_salida, dpi=300)
    plt.show()

def imprimir_matriz_confusion(matriz_confusion: np.ndarray, etiquetas: List[str]) -> None:
    print("\n--- MATRIZ DE CONFUSIÓN POR AUDIO ---")
    print("Filas: Etiqueta real | Columnas: Etiqueta predicha\n")

    print("        ", end="")
    for etiqueta in etiquetas:
        print(f"{str(etiqueta):>8}", end="")
    print()

    for indice, etiqueta_real in enumerate(etiquetas):
        print(f"{str(etiqueta_real):>7} ", end="")
        for columna in range(len(etiquetas)):
            print(f"{matriz_confusion[indice, columna]:>8}", end="")
        print()


def imprimir_resultados_por_audio(resultados: List[ResultadoAudio], limite: int = 150) -> None:
    print("\n--- RESULTADOS POR AUDIO ---")
    print("Cada fila es una predicción final de un archivo completo.\n")

    for resultado in resultados[:limite]:
        estado = "" if resultado["correcto"] else " "
        mejores_distorsiones = resultado.get("mejores_distorsiones", [])
        mejores_distorsiones_texto = ", ".join([f"{etiqueta}:{distorsion:.4f}" for etiqueta, distorsion in mejores_distorsiones])

        print(
            f"{estado} archivo={resultado['archivo']} | "
            f"real={resultado['etiqueta_real']} | "
            f"predicha={resultado['etiqueta_predicha']} | "
            f"marcos={resultado['numero_marcos']} | "
            f"recorte={resultado.get('recorte_inicio')}:{resultado.get('recorte_fin')} | "
            f"mejores=[{mejores_distorsiones_texto}]"
        )

    if len(resultados) > limite:
        print(f"... se omitieron {len(resultados) - limite} resultados más.")


def principal():
    print("=== Punto 6: Matriz de Confusión por Audio ===\n")

    # Configuración principal.
    orden_lpc = 12
    numero_centroides = 32
    numero_audios_entrenamiento = 10

    # Para evaluar todos los audios, incluyendo los de entrenamiento:
    probar_todos_los_audios = False

    if probar_todos_los_audios:
        omitir_primeros_prueba = 0
        numero_audios_prueba = 20
    else:
        omitir_primeros_prueba = 0
        numero_audios_prueba = 10

    print("Procesando audios con recorte por potencia...")
    senales_procesadas = procesar_audios(
        tamano_ventana=160,
        salto_ventana=64,
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
        numero_centroides=numero_centroides,
        numero_audios_entrenamiento=numero_audios_entrenamiento,
        maximo_iteraciones=80,
        epsilon=1e-4,
    )

    print("\nResumen de cuantizadores:")
    for etiqueta, cuantizador in cuantizadores.items():
        print(
            f"  {etiqueta}: "
            f"audios={cuantizador['numero_audios_entrenamiento']} | "
            f"vectores={cuantizador['numero_vectores_entrenamiento']} | "
            f"centroides={cuantizador['numero_centroides_reales']}"
        )

    print("\nExtrayendo LPC/autocorrelaciones de audios de prueba...")
    vectores_prueba = extraer_vectores_prueba(
        senales_procesadas,
        orden=orden_lpc,
        omitir_primeros=omitir_primeros_prueba,
        maximo_archivos=numero_audios_prueba,
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


    
    graficar_matriz_confusion(
        matriz_confusion,
        etiquetas,
        normalizar=False,
        titulo="Matriz de confusión normalizada por etiqueta real",
        ruta_salida="matriz_confusion_entrenamiento.png",
    )

    metricas = calcular_metricas(matriz_confusion)

    print("\n--- MÉTRICAS DE DESEMPEÑO POR AUDIO ---")
    print(f"Audios evaluados: {metricas['total_audios_evaluados']}")
    print(f"Audios correctos: {metricas['correctos']}")
    print(f"Exactitud global: {metricas['exactitud']:.4f}")
    print(f"Precisión promedio: {metricas['precision_promedio']:.4f}")
    print(f"Sensibilidad promedio: {metricas['sensibilidad_promedio']:.4f}")

    print("\n--- MÉTRICAS POR CLASE ---")
    for indice, etiqueta in enumerate(etiquetas):
        print(
            f"{etiqueta}: "
            f"Precision={metricas['precision_por_clase'][indice]:.4f}, "
            f"Sensibilidad={metricas['sensibilidad_por_clase'][indice]:.4f}"
        )

    print("\n Punto 6 completado")
    return matriz_confusion, metricas, resultados


if __name__ == "__main__":
    matriz_confusion, metricas, resultados = principal()

    # python -m src.app.punto_6_confusion > exec.txt