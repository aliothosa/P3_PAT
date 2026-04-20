import numpy as np
from dataclasses import dataclass


@dataclass
class Segmento:
    senalSRC: str
    numeroSegmento: int
    vectorDePotencia: np.ndarray[float]
    matrizDeCorrelacion: np.ndarray[float]
    vectorDeCorrelacion: np.ndarray[float]
    coeficientesLPC: np.ndarray[float]