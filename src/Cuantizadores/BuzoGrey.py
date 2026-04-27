from typing import List

import numpy as np
from src.utils.filtroWiener import filtroWiener
from src.utils.distancias import distanciaEuclidiana, itakuraSaito
from src.data.SegmentoDePotencia import Segmento
import numpy as np
from src.utils.distancias import distanciaEuclidiana
from math import log2
from multiprocessing import Process, Pool

class CuantizadorVectorial:
    def __init__(self, numeroDeCentroides: int, perturbaciones: np.ndarray, umbralDeDistancia: float = 1e-4):
        if ( log2(numeroDeCentroides) % 1 != 0):
            raise ValueError("El número de centroides debe ser una potencia de 2.")
        self.numeroDeCetnroides = numeroDeCentroides
        self.constantesDePerturbacion = perturbaciones
        self.centroides = {}
        self.grupos = {}
        self.distancia_global = 0 
        self.puntos:np.ndarray = None
        self.umbralDeDistancia = umbralDeDistancia
        
    def __encontrar_primer_centroide(self, puntos: np.ndarray) -> np.ndarray:
        return np.mean(puntos, axis=0)
    
    
    def __perturbar_centroides(self) -> float:
        contador = -1
        centroides_nuevos = {}
        for centroide in self.centroides.values():
            centroide_perturbado_0 = centroide * self.constantesDePerturbacion[0]
            centroide_perturbado_1 = centroide * self.constantesDePerturbacion[1]
            
            centroides_nuevos[contador := contador + 1] = centroide_perturbado_0
            centroides_nuevos[contador := contador + 1] = centroide_perturbado_1    
        self.centroides = centroides_nuevos
        
    def  __recalcular_centroides(self) -> None:
        for indice, grupo in self.grupos.items():
            self.centroides[indice] = np.mean(grupo, axis=0)
            
    def agrupar_puntos(self, segmentStart: int, segmentEnd: int) -> tuple[dict[int, list[np.ndarray]], float]:
        grupos = {i: [] for i in self.centroides.keys()}
        distancia_segmento = 0

        for punto in self.puntos[segmentStart:segmentEnd]:
            min_distancia = float("inf")
            indice_cercano = None

            for indice, centroide in self.centroides.items():
                distancia = distanciaEuclidiana(punto, centroide)

                if distancia < min_distancia:
                    min_distancia = distancia
                    indice_cercano = indice

            grupos[indice_cercano].append(punto)
            distancia_segmento += min_distancia

        return grupos, distancia_segmento


    def agrupar_puntos_paralelo(self, numero_procesos: int = 10) -> None:
        total_puntos = len(self.puntos)
        segmento_size = total_puntos // numero_procesos

        segmentos = []

        for i in range(numero_procesos):
            start = i * segmento_size
            end = (i + 1) * segmento_size if i != numero_procesos - 1 else total_puntos
            segmentos.append((start, end))

        with Pool(processes=numero_procesos) as pool:
            resultados = pool.starmap(self.agrupar_puntos, segmentos)

        self.merge_resultados(resultados)


    def merge_resultados(self, resultados: list[tuple[dict[int, list[np.ndarray]], float]]) -> None:
        grupos_finales = {i: [] for i in self.centroides.keys()}
        distancia_global = 0

        for grupos, distancia in resultados:
            distancia_global += distancia

            for indice, puntos in grupos.items():
                grupos_finales[indice].extend(puntos)

        self.grupos = grupos_finales
        self.distancia_global = distancia_global
        
    def entrenar(self, puntos: np.ndarray) -> bool:
        self.centroides[0] = self.__encontrar_primer_centroide(puntos)
        self.puntos = puntos
        
        while len(self.centroides) < self.numeroDeCetnroides:
                self.__perturbar_centroides()

                distancia_anterior = float('inf')

                while True:
                    self.agrupar_puntos_paralelo()
                    self.__recalcular_centroides()
                    print(f"Distancia global: {self.distancia_global}")
                    print(f"Distancia anterior: {distancia_anterior}")
                    if distancia_anterior != float('inf'):
                        cambio = abs(distancia_anterior - self.distancia_global) / distancia_anterior

                        if cambio < self.umbralDeDistancia:
                            break

                    distancia_anterior = self.distancia_global

        return True
        

    def cuantizar(self, punto: np.ndarray) -> int:
        min_distancia = float('inf')
        for indice, centroide in self.centroides.items():
            distancia = distanciaEuclidiana(punto, centroide)
            if distancia < min_distancia:
                min_distancia = distancia
                indice_cercano = indice
        self.grupos[indice_cercano].append(punto)
        self.distancia_global += min_distancia
    
    def obtenerCentroides(self) -> dict[int, np.ndarray]:
        return self.centroides