from typing import List, Tuple

import numpy as np


def funcionDeAutocorrelacionWiener(N: int, ζ: int, señalBase: List[float],  valoresIniciales: List[float]) -> float:
    """
    Calcula autocorrelación/correlación con desplazamiento ζ sin usar índices negativos
    circulares accidentalmente.

    La versión anterior usaba valoresIniciales[i + ζ] cuando ζ era negativo y
    i + ζ < 0. En Python eso accede al final del arreglo, lo cual mete muestras del
    final del audio al inicio y contamina la autocorrelación.

    Para LPC normalmente se manda señalBase == valoresIniciales. En ese caso:
        R(k) = 1/N * sum_{i=0}^{N-k-1} x[i] x[i+k]
    y R(-k) = R(k).
    """
    x = np.asarray(señalBase, dtype=np.float64)
    y = np.asarray(valoresIniciales, dtype=np.float64)

    N = min(int(N), len(x), len(y))
    lag = int(ζ)

    if N <= 0 or abs(lag) >= N:
        return 0.0

    if lag >= 0:
        producto = x[:N - lag] * x[lag:N]
    else:
        k = -lag
        producto = x[k:N] * y[:N - k]

    return float(np.sum(producto) / N)


def procesarMatrizYVectorDeAutocorrelacion(longitud: int,numeroCoeficientes: int,señalBase: List[float],valoresIniciales: List[float])-> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve la matriz de autocorrelación y el vector de autocorrelación necesarios
    para obtener los coeficientes del predictor LPC mediante filtro de Wiener.
    """
    numeroCoeficientes = int(numeroCoeficientes)

    if numeroCoeficientes <= 0:
        raise ValueError("numeroCoeficientes debe ser mayor que cero.")

    mapAutoCorrelaciones = {}

    for i in range(numeroCoeficientes - 1, -numeroCoeficientes - 1, -1):
        mapAutoCorrelaciones[i] = funcionDeAutocorrelacionWiener(
            longitud,
            i,
            señalBase,
            valoresIniciales,
        )

    vectorDeAutoCorrelacion = np.array(
        [mapAutoCorrelaciones[i] for i in range(-1, -numeroCoeficientes - 1, -1)],
        dtype=np.float64,
    )

    matrizDeAutoCorrelacionBase = np.zeros((numeroCoeficientes, numeroCoeficientes), dtype=np.float64)

    for i in range(numeroCoeficientes):
        for j in range(numeroCoeficientes):
            indice = i - j
            matrizDeAutoCorrelacionBase[i, j] = mapAutoCorrelaciones[indice]

    return matrizDeAutoCorrelacionBase, vectorDeAutoCorrelacion


def filtroWiener(señalBase: List[float], valoresIniciales: List[float],numeroCoeficientes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtiene coeficientes LPC usando el sistema de Wiener-Hopf.

    Returns:
        coeficientes, matrizCorrelacion, vectorCorrelacion
    """
    matrizCorrelacion, vectorCorrelacion = procesarMatrizYVectorDeAutocorrelacion(
        len(señalBase),
        numeroCoeficientes,
        señalBase,
        valoresIniciales,
    )

    try:
        coeficientes = np.linalg.solve(matrizCorrelacion, vectorCorrelacion)
    except np.linalg.LinAlgError:
        # Regularización pequeña para marcos con poca energía o matrices casi singulares.
        regularizacion = 1e-8 * np.eye(numeroCoeficientes)
        coeficientes = np.linalg.solve(matrizCorrelacion + regularizacion, vectorCorrelacion)

    return coeficientes, matrizCorrelacion, vectorCorrelacion
