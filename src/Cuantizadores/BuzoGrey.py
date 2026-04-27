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
                
                
    def agrupar_puntos_vectorizado(self) -> None:
        centroides_keys = list(self.centroides.keys())
        centroides_array = np.array([self.centroides[k] for k in centroides_keys])

        distancias = np.linalg.norm(
            self.puntos[:, np.newaxis, :] - centroides_array[np.newaxis, :, :],
            axis=2
        )
        
        indices_minimos = np.argmin(distancias, axis=1)

        distancias_minimas = np.min(distancias, axis=1)

        self.distancia_global = float(np.sum(distancias_minimas))

        self.grupos = {k: [] for k in centroides_keys}

        for punto, indice_array in zip(self.puntos, indices_minimos):
            indice_real = centroides_keys[indice_array]
            self.grupos[indice_real].append(punto)



        
    def entrenar(self, puntos: np.ndarray) -> bool:
        self.centroides[0] = self.__encontrar_primer_centroide(puntos)
        self.puntos = puntos
        
        while len(self.centroides) < self.numeroDeCetnroides:
                self.__perturbar_centroides()

                distancia_anterior = float('inf')

                while True:
                    self.agrupar_puntos_vectorizado()
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
        return indice_cercano
    
    def obtenerCentroides(self) -> dict[int, np.ndarray]:
        return self.centroides