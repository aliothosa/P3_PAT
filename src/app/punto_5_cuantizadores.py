from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np

from src.app.punto_4_hamming_filtro import procesar_audios

# Importar filtroWiener desde utils.
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.filtroWiener import filtroWiener as filtro_wiener  

AudioLPC = Dict[str, object]
Cuantizador = Dict[str, object]

# CONSTANTES
EPSILON_ESTABILIDAD = 1e-12
SIGNO_COEFICIENTES_LPC_EN_FILTRO = -1.0

# Normaliza r por r(0) para que el volumen del audio no domine la distancia.
def normalizar_autocorrelacion(autocorrelacion: np.ndarray) -> np.ndarray:
    autocorrelacion = np.asarray(autocorrelacion, dtype=np.float64).copy()

    if len(autocorrelacion) == 0:
        return autocorrelacion

    if abs(autocorrelacion[0]) > EPSILON_ESTABILIDAD:
        autocorrelacion = autocorrelacion / autocorrelacion[0]

    return autocorrelacion

# Construye r = [r(0), r(1), ..., r(M)] usando la salida de filtroWiener
def extraer_autocorrelacion_desde_wiener(matriz_correlacion: np.ndarray, vector_correlacion: np.ndarray, orden: int) -> np.ndarray:

    matriz_correlacion = np.asarray(matriz_correlacion, dtype=np.float64)
    vector_correlacion = np.asarray(vector_correlacion, dtype=np.float64).reshape(-1)

    autocorrelacion = np.zeros(orden + 1, dtype=np.float64)
    autocorrelacion[0] = matriz_correlacion[0, 0] if matriz_correlacion.size else 0.0

    limite = min(orden, len(vector_correlacion))
    autocorrelacion[1:limite + 1] = vector_correlacion[:limite]

    return normalizar_autocorrelacion(autocorrelacion)

"""
Calcula LPC y conserva la matriz/vector de autocorrelación que devuelve filtroWiener.

Retorna:
    coeficientes_lpc: forma (orden,)
    autocorrelacion_marco: forma (orden + 1,), normalizado por r(0)
    matriz_correlacion: forma (orden, orden)
    vector_correlacion: forma (orden,)
"""
def calcular_lpc_y_correlaciones(marco: np.ndarray, orden: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    marco = np.asarray(marco, dtype=np.float64)

    if len(marco) <= orden or float(np.mean(marco ** 2)) <= EPSILON_ESTABILIDAD:
        return (
            np.zeros(orden, dtype=np.float64),
            np.zeros(orden + 1, dtype=np.float64),
            np.zeros((orden, orden), dtype=np.float64),
            np.zeros(orden, dtype=np.float64),
        )

    try:
        coeficientes, matriz_correlacion, vector_correlacion = filtro_wiener(marco.tolist(), marco.tolist(), orden)
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

    autocorrelacion_marco = extraer_autocorrelacion_desde_wiener(matriz_correlacion, vector_correlacion, orden)

    # Evita que NaN o inf contaminen el cuantizador.
    coeficientes = np.nan_to_num(coeficientes, nan=0.0, posinf=0.0, neginf=0.0)
    autocorrelacion_marco = np.nan_to_num(autocorrelacion_marco, nan=0.0, posinf=0.0, neginf=0.0)

    return coeficientes, autocorrelacion_marco, matriz_correlacion, vector_correlacion


# Alias para conservar compatibilidad con el nombre anterior.
def calcular_coeficientes_lpc(marco: np.ndarray, orden: int = 12) -> np.ndarray:
    coeficientes, _, _, _ = calcular_lpc_y_correlaciones(marco, orden=orden)
    return coeficientes

# Calcula r_a(0), r_a(1), ..., r_a(M) para uno o muchos filtros LPC.
def calcular_autocorrelacion_filtros_lpc(coeficientes_lpc: np.ndarray, signo_coeficientes: float = SIGNO_COEFICIENTES_LPC_EN_FILTRO) -> np.ndarray:
    
    # coeficientes_lpc puede tener forma: (orden,) o (numero_centroides, orden)
    coeficientes_lpc = np.asarray(coeficientes_lpc, dtype=np.float64)
    es_unico = coeficientes_lpc.ndim == 1

    if es_unico:
        coeficientes_lpc = coeficientes_lpc.reshape(1, -1)

    numero_filtros, orden = coeficientes_lpc.shape
    filtros = np.concatenate(
        [np.ones((numero_filtros, 1), dtype=np.float64), signo_coeficientes * coeficientes_lpc],
        axis=1,
    )

    autocorrelaciones_filtros = np.zeros((numero_filtros, orden + 1), dtype=np.float64)

    for desplazamiento in range(orden + 1):
        autocorrelaciones_filtros[:, desplazamiento] = np.sum(
            filtros[:, :orden + 1 - desplazamiento] * filtros[:, desplazamiento:],
            axis=1,
        )

    return autocorrelaciones_filtros[0] if es_unico else autocorrelaciones_filtros

"""
Calcula la distancia de Itakura-Saito de cada marco contra cada centroide.

Usa la fórmula:
    d = r(0) r_a(0) + 2 * sum_{n=1}^{M} r(n) r_a(n)

Retorna:
    matriz de distancias con forma (numero_marcos, numero_centroides)
"""

def calcular_distancias_itakura_saito_autocorrelacion(autocorrelaciones_marcos: np.ndarray, centroides_lpc: np.ndarray) -> np.ndarray:
    autocorrelaciones_marcos = np.asarray(autocorrelaciones_marcos, dtype=np.float64)
    centroides_lpc = np.asarray(centroides_lpc, dtype=np.float64)

    if autocorrelaciones_marcos.ndim == 1:
        autocorrelaciones_marcos = autocorrelaciones_marcos.reshape(1, -1)

    if centroides_lpc.ndim == 1:
        centroides_lpc = centroides_lpc.reshape(1, -1)

    autocorrelaciones_filtros = calcular_autocorrelacion_filtros_lpc(centroides_lpc)
    pesos_autocorrelaciones_filtros = autocorrelaciones_filtros.copy()
    pesos_autocorrelaciones_filtros[:, 1:] *= 2.0

    distancias = autocorrelaciones_marcos @ pesos_autocorrelaciones_filtros.T

    # Por estabilidad numérica, la distancia no debe quedar negativa.
    distancias = np.maximum(distancias, EPSILON_ESTABILIDAD)
    distancias = np.nan_to_num(distancias, nan=1e12, posinf=1e12, neginf=1e12)

    return distancias


# Alias para conservar compatibilidad con el nombre usado en explicaciones anteriores
def calcular_distancia_itakura_saito_desde_autocorrelacion(autocorrelacion_marco: np.ndarray, lpc_candidato: np.ndarray) -> float:
    return float(calcular_distancias_itakura_saito_autocorrelacion(autocorrelacion_marco, lpc_candidato)[0, 0])


"""
Convierte todos los marcos de un audio procesado en LPC y autocorrelaciones.

Retorna:
    lpcs: forma (numero_marcos, orden)
    autocorrelaciones: forma (numero_marcos, orden + 1)
    matrices_correlacion: lista de matrices por marco
    vectores_correlacion: lista de vectores por marco
"""
def extraer_lpc_de_audio(audio_procesado: dict, orden: int = 12) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:

    marcos = audio_procesado["marcos"]

    lpcs = []
    autocorrelaciones = []
    matrices = []
    vectores = []

    for marco in marcos:
        coeficientes, autocorrelacion_marco, matriz, vector = calcular_lpc_y_correlaciones(marco, orden)

        # Saltamos marcos sin energía útil.
        if len(autocorrelacion_marco) == 0 or abs(autocorrelacion_marco[0]) <= EPSILON_ESTABILIDAD:
            continue

        lpcs.append(coeficientes)
        autocorrelaciones.append(autocorrelacion_marco)
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


def extraer_vectores_lpc(senales_por_etiqueta: Dict[str, List[dict]], orden: int = 12, inicio: int = 0, maximo_archivos: Optional[int] = 10) -> Dict[str, List[AudioLPC]]:

    vectores_lpc: Dict[str, List[AudioLPC]] = {}

    for etiqueta, lista_audios in senales_por_etiqueta.items():
        fin = None if maximo_archivos is None else inicio + maximo_archivos
        audios_seleccionados = lista_audios[inicio:fin]
        vectores_lpc[etiqueta] = []

        for audio_procesado in audios_seleccionados:
            lpc_audio, autocorrelacion_audio, matrices_audio, vectores_audio = extraer_lpc_de_audio(audio_procesado, orden=orden)

            vectores_lpc[etiqueta].append(
                {
                    "etiqueta": etiqueta,
                    "archivo": audio_procesado["archivo"],
                    "ruta": audio_procesado["ruta"],
                    "frecuencia_muestreo": audio_procesado["frecuencia_muestreo"],
                    "lpc": lpc_audio,
                    "autocorrelacion": autocorrelacion_audio,
                    "matrices_correlacion": matrices_audio,
                    "vectores_correlacion": vectores_audio,
                    "numero_marcos": len(lpc_audio),
                    "numero_marcos_antes_filtro": audio_procesado.get("numero_marcos_antes_filtro"),
                    "recorte_inicio": audio_procesado.get("recorte_inicio"),
                    "recorte_fin": audio_procesado.get("recorte_fin"),
                    "marcos_descartados": audio_procesado.get("marcos_descartados", 0),
                }
            )

    return vectores_lpc


def validar_potencia_de_dos(valor: int) -> bool:
    return valor >= 1 and (valor & (valor - 1)) == 0

"""
Algoritmo LBG usando distancia Itakura-Saito para asignar marcos.

Los centroides siguen siendo vectores LPC. La asignación se hace comparando la
autocorrelación del marco contra la autocorrelación del filtro LPC del centroide.

"""

def crear_cuantizador_lbg(vectores_lpc: np.ndarray, autocorrelaciones: np.ndarray, numero_centroides: int, maximo_iteraciones: int = 80, epsilon: float = 1e-4, perturbacion: float = 0.01) -> Tuple[np.ndarray, List[float]]:

    vectores_lpc = np.asarray(vectores_lpc, dtype=np.float64)
    autocorrelaciones = np.asarray(autocorrelaciones, dtype=np.float64)

    if len(vectores_lpc) == 0:
        raise ValueError("No hay vectores LPC para entrenar el cuantizador.")

    if len(vectores_lpc) != len(autocorrelaciones):
        raise ValueError("vectores_lpc y autocorrelaciones deben tener la misma cantidad de filas.")

    if not validar_potencia_de_dos(numero_centroides):
        raise ValueError("numero_centroides debe ser potencia de 2: 1, 2, 4, 8, ..., 256.")

    mascara_valida = (
        np.all(np.isfinite(vectores_lpc), axis=1)
        & np.all(np.isfinite(autocorrelaciones), axis=1)
        & (np.abs(autocorrelaciones[:, 0]) > EPSILON_ESTABILIDAD)
    )

    vectores_lpc = vectores_lpc[mascara_valida]
    autocorrelaciones = autocorrelaciones[mascara_valida]

    if len(vectores_lpc) == 0:
        raise ValueError("No quedaron vectores válidos para entrenar el cuantizador.")

    centroides = np.asarray([np.mean(vectores_lpc, axis=0)], dtype=np.float64)
    distorsiones: List[float] = []

    etapas = int(np.log2(numero_centroides))

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

        for _ in range(maximo_iteraciones):
            matriz_distancias = calcular_distancias_itakura_saito_autocorrelacion(autocorrelaciones, centroides)

            asignaciones = np.argmin(matriz_distancias, axis=1)
            distancias_minimas = np.min(matriz_distancias, axis=1)
            distorsion = float(np.mean(distancias_minimas))
            distorsiones.append(distorsion)

            nuevos_centroides = []

            for indice_centroide in range(len(centroides)):
                vectores_asignados = vectores_lpc[asignaciones == indice_centroide]

                if len(vectores_asignados) > 0:
                    nuevos_centroides.append(np.mean(vectores_asignados, axis=0))
                else:
                    nuevos_centroides.append(centroides[indice_centroide])

            nuevos_centroides = np.asarray(nuevos_centroides, dtype=np.float64)

            cambio = float(np.mean(np.linalg.norm(nuevos_centroides - centroides, axis=1)))
            centroides = nuevos_centroides

            if cambio < epsilon:
                break

    return centroides, distorsiones

# Entrena un cuantizador por etiqueta.
def entrenar_cuantizadores(senales_procesadas: Optional[Dict[str, List[dict]]] = None, orden: int = 12, numero_centroides: int = 256, numero_audios_entrenamiento: int = 10, maximo_iteraciones: int = 80, epsilon: float = 1e-4) -> Tuple[Dict[str, List[AudioLPC]], Dict[str, Cuantizador]]:

    if senales_procesadas is None:
        senales_procesadas = procesar_audios()

    vectores_entrenamiento = extraer_vectores_lpc(
        senales_procesadas,
        orden=orden,
        inicio=0,
        maximo_archivos=numero_audios_entrenamiento,
    )

    cuantizadores: Dict[str, Cuantizador] = {}

    for etiqueta, audios_lpc in vectores_entrenamiento.items():
        audios_validos = [informacion_audio for informacion_audio in audios_lpc if len(informacion_audio["lpc"]) > 0]

        if not audios_validos:
            continue

        todos_los_lpc = np.vstack([informacion_audio["lpc"] for informacion_audio in audios_validos])
        todas_las_autocorrelaciones = np.vstack([informacion_audio["autocorrelacion"] for informacion_audio in audios_validos])

        if len(todos_los_lpc) < numero_centroides:
            # Mantener potencia de 2 menor o igual al número de vectores disponibles.
            numero_centroides_reales = 2 ** int(np.floor(np.log2(max(1, len(todos_los_lpc)))))
        else:
            numero_centroides_reales = numero_centroides

        centroides, distorsion = crear_cuantizador_lbg(
            todos_los_lpc,
            todas_las_autocorrelaciones,
            numero_centroides=numero_centroides_reales,
            maximo_iteraciones=maximo_iteraciones,
            epsilon=epsilon,
        )

        cuantizadores[etiqueta] = {
            "etiqueta": etiqueta,
            "centroides": centroides,
            "distorsion_entrenamiento": distorsion,
            "numero_audios_entrenamiento": len(audios_validos),
            "numero_vectores_entrenamiento": len(todos_los_lpc),
            "orden_lpc": orden,
            "numero_centroides_solicitados": numero_centroides,
            "numero_centroides_reales": len(centroides),
            "signo_filtro_lpc": SIGNO_COEFICIENTES_LPC_EN_FILTRO,
        }

    return vectores_entrenamiento, cuantizadores

"""
Promedia eliminando un porcentaje de las distancias más altas.

Sirve para que uno o dos marcos malos no dominen la decisión final del audio.
"""

def media_recortada(valores: np.ndarray, porcentaje_recorte_superior: float = 0.10) -> float:

    valores = np.asarray(valores, dtype=np.float64)

    if len(valores) == 0:
        return float("inf")

    valores = np.sort(valores)
    corte_superior = int(len(valores) * (1.0 - porcentaje_recorte_superior))
    corte_superior = max(1, corte_superior)

    return float(np.mean(valores[:corte_superior]))

"""
Calcula la distorsión de un audio completo contra un cuantizador.
Para cada marco:
    1. Usa r(0), r(1), ..., r(M) del marco.
    2. Calcula distancia Itakura-Saito contra cada centroide LPC.
    3. Toma la distancia mínima.

"""
def calcular_distorsion_audio(autocorrelacion_audio: np.ndarray, centroides_lpc: np.ndarray, porcentaje_recorte_superior: float = 0.10) -> float:

    autocorrelacion_audio = np.asarray(autocorrelacion_audio, dtype=np.float64)

    if len(autocorrelacion_audio) == 0:
        return float("inf")

    matriz_distancias = calcular_distancias_itakura_saito_autocorrelacion(autocorrelacion_audio, centroides_lpc)
    distancias_minimas = np.min(matriz_distancias, axis=1)

    return media_recortada(distancias_minimas, porcentaje_recorte_superior=porcentaje_recorte_superior)

"""
Clasifica un AUDIO COMPLETO.

La predicción es la etiqueta cuyo cuantizador tenga menor distorsión promedio sobre los marcos útiles del audio.
"""

def clasificar_audio_por_distorsion(autocorrelacion_audio: np.ndarray, cuantizadores: Dict[str, Cuantizador], porcentaje_recorte_superior: float = 0.10) -> Tuple[str, Dict[str, float]]:

    distorsiones_por_etiqueta: Dict[str, float] = {}

    for etiqueta, cuantizador in cuantizadores.items():
        distorsiones_por_etiqueta[etiqueta] = calcular_distorsion_audio(
            autocorrelacion_audio,
            cuantizador["centroides"],
            porcentaje_recorte_superior=porcentaje_recorte_superior,
        )

    etiqueta_predicha = min(distorsiones_por_etiqueta, key=distorsiones_por_etiqueta.get)

    return etiqueta_predicha, distorsiones_por_etiqueta


def principal():
    print("=== Punto 5: Cuantizadores Vectoriales con LPC + Itakura-Saito ===\n")

    senales_procesadas = procesar_audios()
    vectores_entrenamiento, cuantizadores = entrenar_cuantizadores(
        senales_procesadas=senales_procesadas,
        orden=12,
        numero_centroides=32,
        numero_audios_entrenamiento=10,
    )

    for etiqueta, cuantizador in cuantizadores.items():
        print(f"\nEtiqueta: {etiqueta}")
        print(f"  Audios entrenamiento: {cuantizador['numero_audios_entrenamiento']}")
        print(f"  Vectores LPC entrenamiento: {cuantizador['numero_vectores_entrenamiento']}")
        print(f"  Centroides generados: {len(cuantizador['centroides'])}")
        print(f"  Distorsión final entrenamiento: {cuantizador['distorsion_entrenamiento'][-1]:.6f}")

    print("\n  Punto 5 completado")
    return vectores_entrenamiento, cuantizadores


if __name__ == "__main__":
    vectores_lpc, cuantizadores = principal()
