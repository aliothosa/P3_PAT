import numpy as np
from typing import Dict, Tuple

# Cargar módulos requeridos
from punto_4_hamming_filtro import senales_procesadas
from punto_5_cuantizadores import calcular_coeficientes_lpc, distancia_itakura_saito, main as main_punto5


def extraer_vectores_prueba(senales_dict: dict, orden: int = 12, skip_first: int = 10, 
                             max_archivos: int = 5) -> dict:
    """
    Extrae vectores LPC de los últimos 5 archivos de cada etiqueta (archivos 11-15).
    
    Args:
        senales_dict: Diccionario con señales procesadas del punto 4
        orden: Orden de los coeficientes LPC
        skip_first: Número de archivos a saltar al inicio (default 10)
        max_archivos: Número de archivos a usar (default 5)
    
    Returns:
        Diccionario con vectores LPC de prueba por etiqueta
    """
    vectores_prueba = {}
    
    for etiqueta, lista_audios in senales_dict.items():
        vectores_prueba[etiqueta] = []
        
        # Usar archivos del 11 al 15 (skip_first:skip_first+max_archivos)
        for audio_procesado in lista_audios[skip_first:skip_first + max_archivos]:
            marcos = audio_procesado["marcos"]
            coeficientes_por_marco = []
            
            for marco in marcos:
                coefs = calcular_coeficientes_lpc(marco, orden)
                coeficientes_por_marco.append(coefs)
            
            vectores_prueba[etiqueta].append(np.array(coeficientes_por_marco))
    
    return vectores_prueba


def clasificar_vector_lpc(vector_lpc: np.ndarray, cuantizadores: dict) -> Tuple[str, float]:
    """
    Clasifica un vector LPC encontrando el centroide más cercano.
    
    Args:
        vector_lpc: Vector LPC a clasificar
        cuantizadores: Diccionario con cuantizadores de cada dígito
    
    Returns:
        Tupla (dígito_clasificado, distancia_mínima)
    """
    min_distancia = float('inf')
    digito_clasificado = None
    
    for digito, cuantizador in cuantizadores.items():
        centroides = cuantizador["centroides"]
        
        # Encontrar distancia mínima a cualquier centroide
        for centroide in centroides:
            dist = distancia_itakura_saito(vector_lpc, centroide)
            if dist < min_distancia:
                min_distancia = dist
                digito_clasificado = digito
    
    return digito_clasificado, min_distancia


def construir_matriz_confusion(vectores_prueba: dict, cuantizadores: dict, 
                               etiquetas: list) -> np.ndarray:
    """
    Construye la matriz de confusión clasificando los vectores de prueba.
    
    Args:
        vectores_prueba: Diccionario con vectores LPC de prueba
        cuantizadores: Diccionario con cuantizadores entrenados
        etiquetas: Lista de etiquetas (dígitos)
    
    Returns:
        Matriz de confusión (real x predicho)
    """
    num_clases = len(etiquetas)
    matriz_confusion = np.zeros((num_clases, num_clases), dtype=int)
    
    # Mapeo de etiqueta a índice
    etiqueta_a_indice = {etiqueta: idx for idx, etiqueta in enumerate(etiquetas)}
    
    for etiqueta_real, lista_vectores_archivos in vectores_prueba.items():
        idx_real = etiqueta_a_indice[etiqueta_real]
        
        for vectores_archivo in lista_vectores_archivos:
            for vector_lpc in vectores_archivo:
                # Clasificar el vector
                etiqueta_predicha, _ = clasificar_vector_lpc(vector_lpc, cuantizadores)
                idx_predicho = etiqueta_a_indice[etiqueta_predicha]
                
                # Incrementar matriz de confusión
                matriz_confusion[idx_real, idx_predicho] += 1
    
    return matriz_confusion


def calcular_metricas(matriz_confusion: np.ndarray) -> dict:
    """
    Calcula métricas de desempeño a partir de la matriz de confusión.
    
    Args:
        matriz_confusion: Matriz de confusión
    
    Returns:
        Diccionario con métricas (accuracy, precision, recall, etc.)
    """
    total = np.sum(matriz_confusion)
    correctos = np.trace(matriz_confusion)
    accuracy = correctos / total if total > 0 else 0
    
    # Precisión y recall por clase
    precision_por_clase = []
    recall_por_clase = []
    
    for i in range(matriz_confusion.shape[0]):
        # Precisión: TP / (TP + FP)
        predichos_clase_i = np.sum(matriz_confusion[:, i])
        tp = matriz_confusion[i, i]
        precision = tp / predichos_clase_i if predichos_clase_i > 0 else 0
        precision_por_clase.append(precision)
        
        # Recall: TP / (TP + FN)
        reales_clase_i = np.sum(matriz_confusion[i, :])
        recall = tp / reales_clase_i if reales_clase_i > 0 else 0
        recall_por_clase.append(recall)
    
    return {
        "accuracy": accuracy,
        "precision_promedio": np.mean(precision_por_clase),
        "recall_promedio": np.mean(recall_por_clase),
        "precision_por_clase": precision_por_clase,
        "recall_por_clase": recall_por_clase
    }


def main():
    print("=== Punto 6: Matriz de Confusión de Reconocimiento ===\n")
    
    # Obtener cuantizadores del punto 5
    print("Cargando cuantizadores del punto 5...")
    _, cuantizadores = main_punto5()
    
    print("\nExtrayendo vectores LPC de prueba (5 archivos por dígito, archivos 11-15)...")
    vectores_prueba = extraer_vectores_prueba(senales_procesadas, orden=12, 
                                               skip_first=10, max_archivos=5)
    
    # Obtener lista de etiquetas
    etiquetas = sorted(list(vectores_prueba.keys()))
    print(f"Etiquetas encontradas: {etiquetas}")
    
    # Construir matriz de confusión
    print("\nClasificando vectores de prueba...")
    matriz_confusion = construir_matriz_confusion(vectores_prueba, cuantizadores, etiquetas)
    
    # Mostrar matriz de confusión
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print("Filas: Etiqueta Real | Columnas: Etiqueta Predicha\n")
    
    # Header
    print("        ", end="")
    for etiqueta in etiquetas:
        print(f"{str(etiqueta):>8}", end="")
    print()
    
    # Datos
    for i, etiqueta_real in enumerate(etiquetas):
        print(f"{etiqueta_real:>7}  ", end="")
        for j in range(len(etiquetas)):
            print(f"{matriz_confusion[i, j]:>8}", end="")
        print()
    
    # Calcular métricas
    print("\n--- MÉTRICAS DE DESEMPEÑO ---")
    metricas = calcular_metricas(matriz_confusion)
    
    print(f"\nAccuracy Global: {metricas['accuracy']:.4f}")
    print(f"Precisión Promedio: {metricas['precision_promedio']:.4f}")
    print(f"Recall Promedio: {metricas['recall_promedio']:.4f}")
    
    print("\n--- MÉTRICAS POR CLASE ---")
    for idx, etiqueta in enumerate(etiquetas):
        print(f"{etiqueta}: Precision={metricas['precision_por_clase'][idx]:.4f}, "
              f"Recall={metricas['recall_por_clase'][idx]:.4f}")
    
    print("\n✓ Punto 6 completado")
    return matriz_confusion, metricas


if __name__ == "__main__":
    matriz_confusion, metricas = main()
