from typing import List

import numpy as np
from src.utils.filtroWiener import filtroWiener
from src.utils.distancias import distanciaEuclidiana, itakuraSaito
from src.data.SegmentoDePotencia import Segmento
import numpy as np
from src.utils.distancias import distanciaEuclidiana


class CuantizadorVectorial:
    def __init__(self,
        numeroDeCentroides: int,
        constantesDePerturbacion: np.ndarray,
        tolerancia_relativa: float = 1e-4,
        max_iter: int = 50,
        tamano_lote: int = 100_000,
        verbose: bool = True
    ):
        self.numeroDeCentroides = numeroDeCentroides
        self.constantesDePerturbacion = np.asarray(constantesDePerturbacion, dtype=np.float64)
        self.tolerancia_relativa = tolerancia_relativa
        self.max_iter = max_iter
        self.tamano_lote = tamano_lote
        self.verbose = verbose
        self.centroides = {}

    def _validar_entrada(self, puntos: np.ndarray) -> np.ndarray:
        puntos = np.asarray(puntos, dtype=np.float64)

        if puntos.ndim != 2:
            raise ValueError("puntos debe ser una matriz de forma (num_muestras, dimension).")

        if self.numeroDeCentroides <= 0:
            raise ValueError("numeroDeCentroides debe ser mayor que 0.")

        if (self.numeroDeCentroides & (self.numeroDeCentroides - 1)) != 0:
            raise ValueError("Para esta implementación, numeroDeCentroides debe ser potencia de 2.")

        dimension = puntos.shape[1]

        if self.constantesDePerturbacion.ndim != 2:
            raise ValueError("constantesDePerturbacion debe ser una matriz de forma (2, dimension).")

        if self.constantesDePerturbacion.shape != (2, dimension):
            raise ValueError(
                f"constantesDePerturbacion debe tener forma (2, {dimension})."
            )

        return puntos

    def _asignar_puntos(self, puntos: np.ndarray, centroides: np.ndarray):
        """
        Asigna cada punto al centroide más cercano.
        Devuelve:
          - etiquetas: arreglo de enteros con el índice del centroide asignado
          - distorsion_global: suma de distancias cuadradas mínimas
        """
        n = puntos.shape[0]
        etiquetas = np.empty(n, dtype=np.int32)
        distorsion_global = 0.0

        for inicio in range(0, n, self.tamano_lote):
            fin = min(inicio + self.tamano_lote, n)
            lote = puntos[inicio:fin]  # (m, d)

            # diff -> (m, k, d)
            diff = lote[:, None, :] - centroides[None, :, :]

            # distancias cuadradas -> (m, k)
            distancias2 = np.einsum("mkd,mkd->mk", diff, diff, optimize=True)

            etiquetas_lote = np.argmin(distancias2, axis=1)
            etiquetas[inicio:fin] = etiquetas_lote

            dist_minimas = distancias2[np.arange(fin - inicio), etiquetas_lote]
            distorsion_global += float(np.sum(dist_minimas))

        return etiquetas, distorsion_global

    def _recalcular_centroides(
        self,
        puntos: np.ndarray,
        etiquetas: np.ndarray,
        centroides_anteriores: np.ndarray
    ) -> np.ndarray:
        nuevos_centroides = centroides_anteriores.copy()
        k = centroides_anteriores.shape[0]

        for i in range(k):
            mascara = (etiquetas == i)
            if np.any(mascara):
                nuevos_centroides[i] = np.mean(puntos[mascara], axis=0)
            else:
                # Si un grupo queda vacío, conservamos el centroide anterior
                nuevos_centroides[i] = centroides_anteriores[i]

        return nuevos_centroides

    def entrenar(self, puntos: np.ndarray) -> bool:
        puntos = self._validar_entrada(puntos)

        e1, e2 = self.constantesDePerturbacion

        # Centroide inicial
        centroides_actuales = np.mean(puntos, axis=0, keepdims=True)

        etapa = 0
        while centroides_actuales.shape[0] < self.numeroDeCentroides:
            etapa += 1

            # Split de centroides
            centroides_actuales = np.vstack([
                centroides_actuales * e1,
                centroides_actuales * e2
            ])

            if self.verbose:
                print(f"\nEtapa {etapa}: entrenando con {centroides_actuales.shape[0]} centroides")

            distorsion_anterior = None

            for iteracion in range(1, self.max_iter + 1):
                etiquetas, distorsion_actual = self._asignar_puntos(puntos, centroides_actuales)
                nuevos_centroides = self._recalcular_centroides(
                    puntos,
                    etiquetas,
                    centroides_actuales
                )

                if distorsion_anterior is not None:
                    mejora_relativa = abs(distorsion_anterior - distorsion_actual) / max(distorsion_anterior, 1e-12)

                    if self.verbose:
                        print(
                            f"  Iter {iteracion:02d} | "
                            f"Distorsión: {distorsion_actual:.6f} | "
                            f"Mejora relativa: {mejora_relativa:.8f}"
                        )

                    centroides_actuales = nuevos_centroides

                    if mejora_relativa < self.tolerancia_relativa:
                        if self.verbose:
                            print(f"  Convergió en {iteracion} iteraciones.")
                        break
                else:
                    if self.verbose:
                        print(f"  Iter {iteracion:02d} | Distorsión inicial: {distorsion_actual:.6f}")
                    centroides_actuales = nuevos_centroides

                distorsion_anterior = distorsion_actual

        self.centroides = {i: c for i, c in enumerate(centroides_actuales)}
        return True

    def cuantizar(self, punto: np.ndarray) -> int:
        if len(self.centroides) == 0:
            raise ValueError("El cuantizador no ha sido entrenado todavía.")

        punto = np.asarray(punto, dtype=np.float64)
        centroides = np.array([self.centroides[i] for i in sorted(self.centroides.keys())])

        diff = centroides - punto
        distancias2 = np.einsum("kd,kd->k", diff, diff, optimize=True)

        return int(np.argmin(distancias2))

    def obtenerCentroides(self) -> dict[int, np.ndarray]:
        return self.centroides
