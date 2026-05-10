import numpy as np

def distanciaEuclidiana(punto1: np.ndarray, punto2: np.ndarray) -> float:
    return np.sqrt(np.sum((punto1 - punto2) ** 2))
                          

