import numpy as np
from typing import List, Tuple
import sys
import importlib.util
from pathlib import Path

# Cargar el módulo 4_hamming_filtro
spec = importlib.util.spec_from_file_location("hamming_filtro", "4_hamming_filtro.py")
hamming_filtro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hamming_filtro)
senales_procesadas = hamming_filtro.senales_procesadas

# Importar filtroWiener desde utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.filtroWiener import filtroWiener


def calcular_coeficientes_lpc(marco: np.ndarray, orden: int = 12) -> np.ndarray:
    """
    Calcula los coeficientes LPC de orden 12 para un marco de audio usando filtro de Wiener.
    
    Args:
        marco: Marco de audio de 160 puntos
        orden: Orden de los coeficientes LPC (default 12)
    
    Returns:
        Array con los coeficientes LPC de orden 12
    """
    if len(marco) <= orden:
        return np.zeros(orden)
    
    # Usar filtroWiener con el marco como señal base
    # y valores iniciales como el mismo marco desplazado
    coeficientes, _, _ = filtroWiener(marco.tolist(), marco.tolist(), orden)
    return np.array(coeficientes)


def distancia_itakura_saito(lpc1: np.ndarray, lpc2: np.ndarray) -> float:
    """
    Calcula la distancia de Itakura-Saito entre dos vectores LPC.
    
    La distancia de Itakura-Saito es:
    D_IS = log(a1^T * R * a1 / a2^T * R * a2)
    
    donde R es la matriz de autocorrelación.
    
    Para simplificar, usamos:
    D_IS = log(sum(a1) / sum(a2)) + sum((a1 - a2)^2 / a2^2)
    
    Args:
        lpc1: Vector LPC 1
        lpc2: Vector LPC 2
    
    Returns:
        Distancia de Itakura-Saito
    """
    # Evitar división por cero
    a2_safe = np.where(lpc2 == 0, 1e-10, lpc2)
    
    # Distancia de Itakura-Saito simplificada
    distancia = np.sum(((lpc1 - lpc2) ** 2) / (a2_safe ** 2 + 1e-10))
    return np.sqrt(distancia)


def extraer_vectores_lpc(senales_dict: dict, orden: int = 12, max_archivos: int = 10) -> dict:
    """
    Extrae vectores LPC de los primeros 10 archivos de cada etiqueta.
    
    Args:
        senales_dict: Diccionario con señales procesadas del punto 4
        orden: Orden de los coeficientes LPC
        max_archivos: Máximo número de archivos por etiqueta (default 10)
    
    Returns:
        Diccionario con vectores LPC por etiqueta
    """
    vectores_lpc = {}
    
    for etiqueta, lista_audios in senales_dict.items():
        vectores_lpc[etiqueta] = []
        
        # Usar solo los primeros 10 archivos
        for idx, audio_procesado in enumerate(lista_audios[:max_archivos]):
            marcos = audio_procesado["marcos"]
            coeficientes_por_marco = []
            
            for marco in marcos:
                coefs = calcular_coeficientes_lpc(marco, orden)
                coeficientes_por_marco.append(coefs)
            
            vectores_lpc[etiqueta].append(np.array(coeficientes_por_marco))
    
    return vectores_lpc


def crear_cuantizador_lbg(vectores: np.ndarray, num_centroides: int, 
                           max_iteraciones: int = 100, epsilon: float = 0.01) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo LBG (Linde-Buzo-Gray) para crear un cuantizador vectorial.
    
    Args:
        vectores: Array de vectores a cuantizar
        num_centroides: Número de centroides a generar
        max_iteraciones: Máximo número de iteraciones
        epsilon: Umbral de convergencia
    
    Returns:
        Tupla (centroides, distorsión_por_iteracion)
    """
    # Inicializar con el centroide de todos los vectores
    centroide_inicial = np.mean(vectores, axis=0)
    centroides = np.array([centroide_inicial])
    distorsiones = []
    
    for iteracion in range(int(np.log2(num_centroides))):
        # Doblar los centroides (perturbar)
        nuevos_centroides = []
        for cent in centroides:
            nuevos_centroides.append(cent * (1 + 0.01))
            nuevos_centroides.append(cent * (1 - 0.01))
        centroides = np.array(nuevos_centroides)
        
        # Iteraciones de Lloyd
        for it in range(max_iteraciones):
            # Asignar cada vector al centroide más cercano
            distancias = np.zeros((len(vectores), len(centroides)))
            for i, vector in enumerate(vectores):
                for j, centroide in enumerate(centroides):
                    distancias[i, j] = distancia_itakura_saito(vector, centroide)
            
            asignaciones = np.argmin(distancias, axis=1)
            
            # Calcular distorsión
            distorsion = np.mean([np.min(distancias[i]) for i in range(len(vectores))])
            distorsiones.append(distorsion)
            
            # Actualizar centroides
            nuevos_centroides = []
            for k in range(len(centroides)):
                vectores_asignados = vectores[asignaciones == k]
                if len(vectores_asignados) > 0:
                    nuevos_centroides.append(np.mean(vectores_asignados, axis=0))
                else:
                    nuevos_centroides.append(centroides[k])
            
            nuevos_centroides = np.array(nuevos_centroides)
            
            # Verificar convergencia
            cambio = np.mean([distancia_itakura_saito(nuevos_centroides[i], centroides[i]) 
                             for i in range(len(centroides))])
            centroides = nuevos_centroides
            
            if cambio < epsilon:
                break
    
    return centroides, distorsiones


def main():
    print("=== Punto 5: Cuantizadores Vectoriales con LPC (10 archivos por dígito) ===\n")
    
    # Extraer vectores LPC del punto 4 (solo 10 archivos por etiqueta)
    print("Extrayendo vectores LPC de orden 12 (10 archivos por dígito)...")
    vectores_lpc = extraer_vectores_lpc(senales_procesadas, orden=12, max_archivos=10)
    
    # Mostrar información
    for etiqueta, lista_lpc in vectores_lpc.items():
        print(f"\nEtiqueta: {etiqueta}")
        print(f"  Archivos usados: {len(lista_lpc)} / 10")
        if len(lista_lpc) > 0:
            print(f"  Marcos por archivo: {lista_lpc[0].shape[0]}")
            print(f"  Dimensión LPC: {lista_lpc[0].shape[1]}")
    
    # Crear cuantizadores para cada número
    print("\n\nCreando cuantizadores vectoriales (LBG)...")
    cuantizadores = {}
    
    for etiqueta, lista_lpc in vectores_lpc.items():
        print(f"\nProcesando {etiqueta}...")
        
        # Combinar todos los vectores LPC de este dígito
        todos_vectores = np.vstack(lista_lpc)
        
        # Crear cuantizador con 256 centroides
        centroides, distorsion = crear_cuantizador_lbg(todos_vectores, num_centroides=256)
        cuantizadores[etiqueta] = {
            "centroides": centroides,
            "distorsion": distorsion,
            "num_vectores": len(todos_vectores)
        }
        
        print(f"  Centroides generados: {len(centroides)}")
        print(f"  Distorsión final: {distorsion[-1]:.6f}")
    
    print("\n✓ Punto 5 completado")
    return vectores_lpc, cuantizadores


if __name__ == "__main__":
    vectores_lpc, cuantizadores = main()
