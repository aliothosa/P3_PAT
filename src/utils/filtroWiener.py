from typing import List, Tuple
import numpy as np

def funcionDeAutocorrelacionWiener(N: int, ζ: float, señalBase: List[float], valoresIniciales: List[float]) -> float:
    """
        funcion que calcula la autocorrelacion de una señal base con un desplazamiento ζ, utilizando la formula:
        R(ζ) = 1/N * sumatoria(i=0 hasta N-1) de x(i)*x(i+ζ)
        donde x(i) es la señal base, y x(i+ζ) es la señal base desplazada ζ posiciones.
        
        Args:
            N: numero de muestras.
            ζ: desplazamiento para el cual se quiere calcular la autocorrelacion.
            señalBase: la señal base sobre la cual se calculará la autocorrelacion.
            valoresIniciales: los valores iniciales de la función que se quiere aproximar
        Returns:
            El valor de la autocorrelacion para el desplazamiento ζ.
    """
    result = 0
    for i in range(N):
        if i + ζ > N-1:
            break
        
        if (i + ζ) < 0:
            result += 1/N * señalBase[i]*valoresIniciales[i+ζ]
        else:
            result += 1/N * señalBase[i]*señalBase[i+ζ]
    return result
    

def procesarMatrizYVectorDeAutocorrelacion(longitud: int, numeroCoeficientes: int, señalBase: List[float], valoresIniciales: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """ 
        Función que devuelve la matriz de autocorrelación y el vector de autocorrelación
        necesarios para obtener los coeficientes utilizando el filtro de wiener.
    
    Args:
        longitud: numero de muestas.
        numeroCoeficientes: numero de coeficientes que se quieren obtener.
        señalBase: La señal base sobre la cual se calculará la autocorrelación.
        valoresIniciales: Los valores iniciales de la función que se quiere aproximar.
    Returns:
        Una tupla con la matriz de autocorrelación y el vector de autocorrelación representados como numpy arrays.
    """
    mapAutoCorrelaciones = {}
    
    
    for i in range(numeroCoeficientes-1 , -numeroCoeficientes-1, -1):
        mapAutoCorrelaciones[i] = funcionDeAutocorrelacionWiener(longitud, i, señalBase, valoresIniciales)

    vectorDeAutoCorrelacion = np.array([mapAutoCorrelaciones[i] for i in range(-1 , -numeroCoeficientes-1, -1)])
    matrizDeAutoCorrelacionBase = np.zeros((numeroCoeficientes, numeroCoeficientes))
    
    for i in range(numeroCoeficientes):
        for j in range(numeroCoeficientes):
            indice = i-j
            matrizDeAutoCorrelacionBase[i][j] = mapAutoCorrelaciones[indice]
        
    return matrizDeAutoCorrelacionBase, vectorDeAutoCorrelacion
        
def filtroWiener( señalBase : List[float], valoresIniciales : List[float], numeroCoeficientes : int ) -> List[float]:
    """
    Funcion que obtiene los coeficientes utilizando el filtro de wiener de una señal 
        
    Args:
        señalBase: La señal base sobre la cual se calculará la autocorrelación.
        valoresIniciales: Los valores iniciales de la función que se quiere aproximar.
        numeroCoeficientes: El número de coeficientes que se quieren obtener, es decir, k en la ecuación anterior.
    
    Returns:
        Una lista de tamaño coeficientNumber con los coeficientes obtenidos utilizando el filtro de wiener.
    """
    matrizCorrelacion, vectorCorrelacion = procesarMatrizYVectorDeAutocorrelacion(len(señalBase), numeroCoeficientes, señalBase, valoresIniciales)
    
    return np.linalg.solve(matrizCorrelacion, vectorCorrelacion), matrizCorrelacion, vectorCorrelacion