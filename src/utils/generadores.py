import numpy as np

def generadorDeTuplas(limiteInferior :float = 0.0, limiteSuperior: float = 100.0) -> tuple:
    return np.array([np.random.uniform(limiteInferior, limiteSuperior), np.random.uniform(limiteInferior, limiteSuperior)])
