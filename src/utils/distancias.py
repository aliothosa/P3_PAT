import numpy as np
from src.data.SegmentoDePotencia import Segmento

def distanciaEuclidiana(punto1: np.ndarray, punto2: np.ndarray) -> float:
    return np.sqrt(np.sum((punto1 - punto2) ** 2))
                          

def itakuraSaito(seg_test: Segmento, seg_modelo: Segmento) -> float:
    # 1. Obtener r(n) del segmento de prueba
    r = seg_test.vectorDeCorrelacion
    
    # 2. Obtener coeficientes LPC del modelo (a)
    # Nota: Los coeficientes LPC suelen venir como [a1, a2, ..., aM].
    # Debemos incluir el término a0 = 1 al principio.
    lpc = seg_modelo.coeficientesLPC
    a = np.insert(lpc, 0, 1.0) 
    M = len(lpc)
    
    # 3. Calcular ra(n): Autocorrelación de los coeficientes LPC
    # ra(i) = sum_{k=0}^{M-i} (a_k * a_{k+i})
    ra = np.zeros(M + 1)
    for i in range(M + 1):
        for k in range(M + 1 - i):
            ra[i] += a[k] * a[k + i]
            
    # 4. Aplicar la fórmula de la diapositiva:
    # d_is = r(0)ra(0) + 2 * sum_{n=1}^{M} (r(n) * ra(n))
    
    # Término inicial n=0
    distancia = r[0] * ra[0]
    
    # Sumatoria para n=1 hasta M
    suma_intermedia = np.sum(r[1:M+1] * ra[1:M+1])
    
    distancia += 2 * suma_intermedia
    
    return distancia