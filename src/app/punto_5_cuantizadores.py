"""
Punto 5 - LPC y cuantizadores vectoriales LBG.

Cambios importantes:
- Ya no se descartan la matriz y el vector que devuelve filtroWiener.
- Se guarda r(0), r(1), ..., r(M) por cada marco para usar la distancia tipo
  Itakura-Saito indicada por la fórmula:

      d_is = r(0) r_a(0) + 2 * sum_{n=1}^{M} r(n) r_a(n)

- El LBG asigna marcos a centroides usando esa distancia, no una distancia
  cuadrática normalizada entre coeficientes LPC.
- La clasificación de un audio completo se decide con la distorsión promedio
  recortada de todos sus marcos.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np

from src.app.punto_4_hamming_filtro import procesar_audios

# Importar filtroWiener desde utils.
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.filtroWiener import filtroWiener  # noqa: E402

AudioLPC = Dict[str, object]
Cuantizador = Dict[str, object]

EPS = 1e-12

# Si tus coeficientes LPC vienen como predictor:
#   x[n] ≈ a1*x[n-1] + a2*x[n-2] + ...
# el filtro de error es:
#   A(z) = 1 - a1 z^-1 - a2 z^-2 - ...
# Por eso el signo por defecto es -1.0.
# Si el resultado empeora, cambia este valor a +1.0 y vuelve a correr.
SIGNO_COEFICIENTES_LPC_EN_FILTRO = -1.0


def normalizar_autocorrelacion(r: np.ndarray) -> np.ndarray:
    """
    Normaliza r por r(0) para que el volumen del audio no domine la distancia.
    """
    r = np.asarray(r, dtype=np.float64).copy()

    if len(r) == 0:
        return r

    if abs(r[0]) > EPS:
        r = r / r[0]

    return r


def extraer_r_desde_wiener(
    matriz_correlacion: np.ndarray,
    vector_correlacion: np.ndarray,
    orden: int,
) -> np.ndarray:
    """
    Construye r = [r(0), r(1), ..., r(M)] usando la salida de filtroWiener.

    En tu implementación:
    - matriz_correlacion[0, 0] contiene r(0)
    - vector_correlacion contiene los lags usados para resolver el predictor.

    Como autocorrelación es simétrica para la misma señal, usamos esos valores como
    r(1), ..., r(M).
    """
    matriz_correlacion = np.asarray(matriz_correlacion, dtype=np.float64)
    vector_correlacion = np.asarray(vector_correlacion, dtype=np.float64).reshape(-1)

    r = np.zeros(orden + 1, dtype=np.float64)
    r[0] = matriz_correlacion[0, 0] if matriz_correlacion.size else 0.0

    limite = min(orden, len(vector_correlacion))
    r[1:limite + 1] = vector_correlacion[:limite]

    return normalizar_autocorrelacion(r)


def calcular_lpc_y_correlaciones(
    marco: np.ndarray,
    orden: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula LPC y conserva la matriz/vector de autocorrelación que devuelve filtroWiener.

    Returns:
        coeficientes_lpc: forma (orden,)
        r_marco: forma (orden + 1,), normalizado por r(0)
        matriz_correlacion: forma (orden, orden)
        vector_correlacion: forma (orden,)
    """
    marco = np.asarray(marco, dtype=np.float64)

    if len(marco) <= orden or float(np.mean(marco ** 2)) <= EPS:
        return (
            np.zeros(orden, dtype=np.float64),
            np.zeros(orden + 1, dtype=np.float64),
            np.zeros((orden, orden), dtype=np.float64),
            np.zeros(orden, dtype=np.float64),
        )

    try:
        coeficientes, matriz_correlacion, vector_correlacion = filtroWiener(
            marco.tolist(),
            marco.tolist(),
            orden,
        )
    except np.linalg.LinAlgError:
        return (
            np.zeros(orden, dtype=np.float64),
            np.zeros(orden + 1, dtype=np.float64),
            np.zeros((orden, orden), dtype=np.float64),
            np.zeros(orden, dtype=np.float64),
        )

    coeficientes = np.asarray(coeficientes, dtype=np.float64).reshape(-1)
    matriz_correlacion = np.asarray(matriz_correlacion, dtype=np.float64)
    vector_correlacion = np.asarray(vector_correlacion, dtype=np.float64).reshape(-1)

    if len(coeficientes) != orden:
        coeficientes_ajustados = np.zeros(orden, dtype=np.float64)
        limite = min(orden, len(coeficientes))
        coeficientes_ajustados[:limite] = coeficientes[:limite]
        coeficientes = coeficientes_ajustados

    r_marco = extraer_r_desde_wiener(matriz_correlacion, vector_correlacion, orden)

    # Evita que NaN o inf contaminen el cuantizador.
    coeficientes = np.nan_to_num(coeficientes, nan=0.0, posinf=0.0, neginf=0.0)
    r_marco = np.nan_to_num(r_marco, nan=0.0, posinf=0.0, neginf=0.0)

    return coeficientes, r_marco, matriz_correlacion, vector_correlacion


# Alias para conservar compatibilidad con el nombre anterior.
def calcular_coeficientes_lpc(marco: np.ndarray, orden: int = 12) -> np.ndarray:
    coeficientes, _, _, _ = calcular_lpc_y_correlaciones(marco, orden=orden)
    return coeficientes


def autocorrelacion_filtros_lpc(
    coeficientes_lpc: np.ndarray,
    signo_coeficientes: float = SIGNO_COEFICIENTES_LPC_EN_FILTRO,
) -> np.ndarray:
    """
    Calcula r_a(0), r_a(1), ..., r_a(M) para uno o muchos filtros LPC.

    coeficientes_lpc puede tener forma:
        (orden,) o (num_centroides, orden)
    """
    coeficientes_lpc = np.asarray(coeficientes_lpc, dtype=np.float64)
    es_unico = coeficientes_lpc.ndim == 1

    if es_unico:
        coeficientes_lpc = coeficientes_lpc.reshape(1, -1)

    num_filtros, orden = coeficientes_lpc.shape
    filtros = np.concatenate(
        [np.ones((num_filtros, 1), dtype=np.float64), signo_coeficientes * coeficientes_lpc],
        axis=1,
    )

    ra = np.zeros((num_filtros, orden + 1), dtype=np.float64)

    for n in range(orden + 1):
        ra[:, n] = np.sum(filtros[:, :orden + 1 - n] * filtros[:, n:], axis=1)

    return ra[0] if es_unico else ra


def distancias_itakura_saito_autocorr(
    autocorrelaciones_marcos: np.ndarray,
    centroides_lpc: np.ndarray,
) -> np.ndarray:
    """
    Calcula la distancia de Itakura-Saito de cada marco contra cada centroide.

    Usa la fórmula:
        d = r(0) r_a(0) + 2 * sum_{n=1}^{M} r(n) r_a(n)

    Args:
        autocorrelaciones_marcos: forma (num_marcos, orden + 1)
        centroides_lpc: forma (num_centroides, orden)

    Returns:
        matriz de distancias con forma (num_marcos, num_centroides)
    """
    autocorrelaciones_marcos = np.asarray(autocorrelaciones_marcos, dtype=np.float64)
    centroides_lpc = np.asarray(centroides_lpc, dtype=np.float64)

    if autocorrelaciones_marcos.ndim == 1:
        autocorrelaciones_marcos = autocorrelaciones_marcos.reshape(1, -1)

    if centroides_lpc.ndim == 1:
        centroides_lpc = centroides_lpc.reshape(1, -1)

    ra = autocorrelacion_filtros_lpc(centroides_lpc)
    pesos_ra = ra.copy()
    pesos_ra[:, 1:] *= 2.0

    distancias = autocorrelaciones_marcos @ pesos_ra.T

    # Por estabilidad numérica, la distancia no debe quedar negativa.
    distancias = np.maximum(distancias, EPS)
    distancias = np.nan_to_num(distancias, nan=1e12, posinf=1e12, neginf=1e12)

    return distancias


# Alias para conservar compatibilidad con el nombre usado en explicaciones anteriores.
def distancia_itakura_saito_desde_r(r_marco: np.ndarray, lpc_candidato: np.ndarray) -> float:
    return float(distancias_itakura_saito_autocorr(r_marco, lpc_candidato)[0, 0])


def extraer_lpc_de_audio(audio_procesado: dict, orden: int = 12) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Convierte todos los marcos de un audio procesado en LPC y autocorrelaciones.

    Returns:
        lpcs: forma (num_marcos, orden)
        autocorrelaciones: forma (num_marcos, orden + 1)
        matrices_correlacion: lista de matrices por marco
        vectores_correlacion: lista de vectores por marco
    """
    marcos = audio_procesado["marcos"]

    lpcs = []
    autocorrelaciones = []
    matrices = []
    vectores = []

    for marco in marcos:
        coefs, r_marco, matriz, vector = calcular_lpc_y_correlaciones(marco, orden)

        # Saltamos marcos sin energía útil.
        if len(r_marco) == 0 or abs(r_marco[0]) <= EPS:
            continue

        lpcs.append(coefs)
        autocorrelaciones.append(r_marco)
        matrices.append(matriz)
        vectores.append(vector)

    if not lpcs:
        return (
            np.zeros((0, orden), dtype=np.float64),
            np.zeros((0, orden + 1), dtype=np.float64),
            [],
            [],
        )

    return (
        np.asarray(lpcs, dtype=np.float64),
        np.asarray(autocorrelaciones, dtype=np.float64),
        matrices,
        vectores,
    )


def extraer_vectores_lpc(
    senales_dict: Dict[str, List[dict]],
    orden: int = 12,
    inicio: int = 0,
    max_archivos: Optional[int] = 10,
) -> Dict[str, List[AudioLPC]]:
    """
    Extrae LPC por audio conservando metadatos.
    """
    vectores_lpc: Dict[str, List[AudioLPC]] = {}

    for etiqueta, lista_audios in senales_dict.items():
        fin = None if max_archivos is None else inicio + max_archivos
        audios_seleccionados = lista_audios[inicio:fin]
        vectores_lpc[etiqueta] = []

        for audio_procesado in audios_seleccionados:
            lpc_audio, autocorr_audio, matrices_audio, vectores_audio = extraer_lpc_de_audio(
                audio_procesado,
                orden=orden,
            )

            vectores_lpc[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": audio_procesado["archivo"],
                    "ruta": audio_procesado["ruta"],
                    "sr": audio_procesado["sr"],
                    "lpc": lpc_audio,
                    "autocorr": autocorr_audio,
                    "matrices_correlacion": matrices_audio,
                    "vectores_correlacion": vectores_audio,
                    "num_marcos": len(lpc_audio),
                    "num_marcos_antes_filtro": audio_procesado.get("num_marcos_antes_filtro"),
                    "recorte_inicio": audio_procesado.get("recorte_inicio"),
                    "recorte_fin": audio_procesado.get("recorte_fin"),
                    "marcos_descartados": audio_procesado.get("marcos_descartados", 0),
                }
            )

    return vectores_lpc


def validar_potencia_de_dos(valor: int) -> bool:
    return valor >= 1 and (valor & (valor - 1)) == 0


def crear_cuantizador_lbg(
    vectores_lpc: np.ndarray,
    autocorrelaciones: np.ndarray,
    num_centroides: int,
    max_iteraciones: int = 80,
    epsilon: float = 1e-4,
    perturbacion: float = 0.01,
) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo LBG usando distancia Itakura-Saito para asignar marcos.

    Los centroides siguen siendo vectores LPC. La asignación se hace comparando la
    autocorrelación del marco contra la autocorrelación del filtro LPC del centroide.
    """
    vectores_lpc = np.asarray(vectores_lpc, dtype=np.float64)
    autocorrelaciones = np.asarray(autocorrelaciones, dtype=np.float64)

    if len(vectores_lpc) == 0:
        raise ValueError("No hay vectores LPC para entrenar el cuantizador.")

    if len(vectores_lpc) != len(autocorrelaciones):
        raise ValueError("vectores_lpc y autocorrelaciones deben tener la misma cantidad de filas.")

    if not validar_potencia_de_dos(num_centroides):
        raise ValueError("num_centroides debe ser potencia de 2: 1, 2, 4, 8, ..., 256.")

    mascara_valida = (
        np.all(np.isfinite(vectores_lpc), axis=1)
        & np.all(np.isfinite(autocorrelaciones), axis=1)
        & (np.abs(autocorrelaciones[:, 0]) > EPS)
    )

    vectores_lpc = vectores_lpc[mascara_valida]
    autocorrelaciones = autocorrelaciones[mascara_valida]

    if len(vectores_lpc) == 0:
        raise ValueError("No quedaron vectores válidos para entrenar el cuantizador.")

    centroides = np.asarray([np.mean(vectores_lpc, axis=0)], dtype=np.float64)
    distorsiones: List[float] = []

    etapas = int(np.log2(num_centroides))

    for _ in range(etapas):
        centroides = np.asarray(
            [
                nuevo
                for centroide in centroides
                for nuevo in (
                    centroide * (1.0 + perturbacion),
                    centroide * (1.0 - perturbacion),
                )
            ],
            dtype=np.float64,
        )

        for _ in range(max_iteraciones):
            matriz_distancias = distancias_itakura_saito_autocorr(
                autocorrelaciones,
                centroides,
            )

            asignaciones = np.argmin(matriz_distancias, axis=1)
            distancias_minimas = np.min(matriz_distancias, axis=1)
            distorsion = float(np.mean(distancias_minimas))
            distorsiones.append(distorsion)

            nuevos_centroides = []

            for k in range(len(centroides)):
                vectores_asignados = vectores_lpc[asignaciones == k]

                if len(vectores_asignados) > 0:
                    nuevos_centroides.append(np.mean(vectores_asignados, axis=0))
                else:
                    nuevos_centroides.append(centroides[k])

            nuevos_centroides = np.asarray(nuevos_centroides, dtype=np.float64)

            cambio = float(np.mean(np.linalg.norm(nuevos_centroides - centroides, axis=1)))
            centroides = nuevos_centroides

            if cambio < epsilon:
                break

    return centroides, distorsiones


def entrenar_cuantizadores(
    senales_procesadas: Optional[Dict[str, List[dict]]] = None,
    orden: int = 12,
    num_centroides: int = 256,
    num_audios_entrenamiento: int = 10,
    max_iteraciones: int = 80,
    epsilon: float = 1e-4,
) -> Tuple[Dict[str, List[AudioLPC]], Dict[str, Cuantizador]]:
    """
    Entrena un cuantizador por etiqueta.
    """
    if senales_procesadas is None:
        senales_procesadas = procesar_audios()

    vectores_entrenamiento = extraer_vectores_lpc(
        senales_procesadas,
        orden=orden,
        inicio=0,
        max_archivos=num_audios_entrenamiento,
    )

    cuantizadores: Dict[str, Cuantizador] = {}

    for etiqueta, audios_lpc in vectores_entrenamiento.items():
        audios_validos = [audio_info for audio_info in audios_lpc if len(audio_info["lpc"]) > 0]

        if not audios_validos:
            continue

        todos_lpc = np.vstack([audio_info["lpc"] for audio_info in audios_validos])
        todas_autocorr = np.vstack([audio_info["autocorr"] for audio_info in audios_validos])

        if len(todos_lpc) < num_centroides:
            # Mantener potencia de 2 menor o igual al número de vectores disponibles.
            num_centroides_reales = 2 ** int(np.floor(np.log2(max(1, len(todos_lpc)))))
        else:
            num_centroides_reales = num_centroides

        centroides, distorsion = crear_cuantizador_lbg(
            todos_lpc,
            todas_autocorr,
            num_centroides=num_centroides_reales,
            max_iteraciones=max_iteraciones,
            epsilon=epsilon,
        )

        cuantizadores[etiqueta] = {
            "etiqueta": etiqueta,
            "centroides": centroides,
            "distorsion_entrenamiento": distorsion,
            "num_audios_entrenamiento": len(audios_validos),
            "num_vectores_entrenamiento": len(todos_lpc),
            "orden_lpc": orden,
            "num_centroides_solicitados": num_centroides,
            "num_centroides_reales": len(centroides),
            "signo_filtro_lpc": SIGNO_COEFICIENTES_LPC_EN_FILTRO,
        }

    return vectores_entrenamiento, cuantizadores


def media_recortada(valores: np.ndarray, porcentaje_recorte_superior: float = 0.10) -> float:
    """
    Promedia eliminando un porcentaje de las distancias más altas.

    Sirve para que uno o dos marcos malos no dominen la decisión final del audio.
    """
    valores = np.asarray(valores, dtype=np.float64)

    if len(valores) == 0:
        return float("inf")

    valores = np.sort(valores)
    corte_superior = int(len(valores) * (1.0 - porcentaje_recorte_superior))
    corte_superior = max(1, corte_superior)

    return float(np.mean(valores[:corte_superior]))


def calcular_distorsion_audio(
    autocorr_audio: np.ndarray,
    centroides_lpc: np.ndarray,
    porcentaje_recorte_superior: float = 0.10,
) -> float:
    """
    Calcula la distorsión de un audio completo contra un cuantizador.

    Para cada marco:
        1. Usa r(0), r(1), ..., r(M) del marco.
        2. Calcula distancia Itakura-Saito contra cada centroide LPC.
        3. Toma la distancia mínima.

    La distorsión final del audio es una media recortada de esas distancias mínimas.
    """
    autocorr_audio = np.asarray(autocorr_audio, dtype=np.float64)

    if len(autocorr_audio) == 0:
        return float("inf")

    matriz_distancias = distancias_itakura_saito_autocorr(autocorr_audio, centroides_lpc)
    distancias_minimas = np.min(matriz_distancias, axis=1)

    return media_recortada(distancias_minimas, porcentaje_recorte_superior=porcentaje_recorte_superior)


def clasificar_audio_por_distorsion(
    autocorr_audio: np.ndarray,
    cuantizadores: Dict[str, Cuantizador],
    porcentaje_recorte_superior: float = 0.10,
) -> Tuple[str, Dict[str, float]]:
    """
    Clasifica un AUDIO COMPLETO.

    La predicción es la etiqueta cuyo cuantizador tenga menor distorsión promedio
    sobre los marcos útiles del audio.
    """
    distorsiones_por_etiqueta: Dict[str, float] = {}

    for etiqueta, cuantizador in cuantizadores.items():
        distorsiones_por_etiqueta[etiqueta] = calcular_distorsion_audio(
            autocorr_audio,
            cuantizador["centroides"],
            porcentaje_recorte_superior=porcentaje_recorte_superior,
        )

    etiqueta_predicha = min(distorsiones_por_etiqueta, key=distorsiones_por_etiqueta.get)

    return etiqueta_predicha, distorsiones_por_etiqueta


def main():
    print("=== Punto 5: Cuantizadores Vectoriales con LPC + Itakura-Saito ===\n")

    senales_procesadas = procesar_audios()
    vectores_entrenamiento, cuantizadores = entrenar_cuantizadores(
        senales_procesadas=senales_procesadas,
        orden=12,
        num_centroides=256,
        num_audios_entrenamiento=10,
    )

    for etiqueta, cuantizador in cuantizadores.items():
        print(f"\nEtiqueta: {etiqueta}")
        print(f"  Audios entrenamiento: {cuantizador['num_audios_entrenamiento']}")
        print(f"  Vectores LPC entrenamiento: {cuantizador['num_vectores_entrenamiento']}")
        print(f"  Centroides generados: {len(cuantizador['centroides'])}")
        print(f"  Distorsión final entrenamiento: {cuantizador['distorsion_entrenamiento'][-1]:.6f}")

    print("\n✓ Punto 5 completado")
    return vectores_entrenamiento, cuantizadores


if __name__ == "__main__":
    vectores_lpc, cuantizadores = main()
