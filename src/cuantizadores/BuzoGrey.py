import os
from typing import List

import numpy as np
from src.utils.filtroWiener import filtroWiener
from src.utils.distancias import distanciaEuclidiana, itakuraSaito
from src.data.SegmentoDePotencia import Segmento
import numpy as np
from src.utils.distancias import distanciaEuclidiana
from math import log2
from multiprocessing import Process, Pool
from typing import Dict
from pathlib import Path
import pickle
from copy import copy


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class CuantizadorVectorial:
    
    @classmethod
    def load(cls, srcPath: str) -> CuantizadorVectorial:
        archivoObjeto = Path(srcPath)

        if not archivoObjeto.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {srcPath}")

        if not srcPath.endswith(".obj"):
            raise ValueError(f"El archivo debe tener extensión .obj: {srcPath}")

        with open(archivoObjeto, "rb") as archivo:
            obj = pickle.load(archivo)

        if not isinstance(obj, cls):
            raise TypeError("El archivo no contiene un CuantizadorVectorial válido.")

        return obj

    @classmethod
    def write(cls, obj: CuantizadorVectorial, destPath: str) -> None:
        if not destPath.endswith(".obj"):
            raise ValueError(f"El archivo debe tener extensión .obj: {destPath}")

        archivoObjeto = Path(destPath)

        # Copia para no modificar el objeto original
        obj_a_guardar = copy(obj)

        # Limpieza de variables pesadas
        obj_a_guardar.puntos = None

        with open(archivoObjeto, "wb") as archivo:
            pickle.dump(obj_a_guardar, archivo)
            
            
        
    def __init__(self, numeroDeCentroides: int, perturbaciones: np.ndarray, umbralDeDistancia: float = 1e-3):
        if ( log2(numeroDeCentroides) % 1 != 0):
            raise ValueError("El número de centroides debe ser una potencia de 2.")
        self.numeroDeCentroides : int = numeroDeCentroides
        self.constantesDePerturbacion : np.ndarray = perturbaciones
        self.centroides : Dict[int, np.ndarray] = {} # Variable importante para escribir
        self.distancia_global : float = 0 
        self.puntos : np.ndarray = None
        self.umbralDeDistancia : float = umbralDeDistancia    
        
        
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
        
    def  __recalcular_centroides(self, grupos : Dict[int, List[np.ndarray]]) -> None:
        for indice, grupo in grupos.items():
            self.centroides[indice] = np.mean(grupo, axis=0)
                
                
    def agrupar_puntos_vectorizado(self) -> Dict[int, List[np.ndarray]]:
        centroides_keys = list(self.centroides.keys())
        centroides_array = np.array([self.centroides[k] for k in centroides_keys])

        distancias = np.linalg.norm(
            self.puntos[:, np.newaxis, :] - centroides_array[np.newaxis, :, :],
            axis=2
        )
        
        indices_minimos = np.argmin(distancias, axis=1)

        distancias_minimas = np.min(distancias, axis=1)

        self.distancia_global = float(np.sum(distancias_minimas))

        grupos = {k: [] for k in centroides_keys}

        for punto, indice_array in zip(self.puntos, indices_minimos):
            indice_real = centroides_keys[indice_array]
            grupos[indice_real].append(punto)

        return grupos

        
    def entrenar(self, puntos: np.ndarray) -> bool:
        self.centroides[0] = self.__encontrar_primer_centroide(puntos)
        self.puntos = puntos
        distancias_log = []
        while len(self.centroides) < self.numeroDeCentroides:
                self.__perturbar_centroides()

                distancia_anterior = float('inf')

                while True:
                    grupos = self.agrupar_puntos_vectorizado()
                    self.__recalcular_centroides(grupos)
                    
                    if distancia_anterior != float('inf'):
                        cambio = abs(distancia_anterior - self.distancia_global) / distancia_anterior

                        if cambio < self.umbralDeDistancia:
                            break

                    distancias_log.append(self.distancia_global)
                    distancia_anterior = self.distancia_global

        self.write_log(distancias_log)
        return True
    
    def write_log(self, distancias: list[float]) -> None:
        with open(os.path.join(LOG_DIR, f"log_{self.numeroDeCentroides}.csv"), 'a') as f:
            f.write(f"{self.distancia_global}\n")
            for distancia in distancias:
                f.write(f"{distancia}\n")
        

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
    
    
    # Cuantizador de LBG con distancias de Itakura-Saito